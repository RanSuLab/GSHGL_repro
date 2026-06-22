"""
Multi-scale gene-supervised contrastive pretraining.

Supports an optional auxiliary classification objective alongside contrastive learning.
"""

import os
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.amp import autocast, GradScaler

from configs.config import PretrainConfig
from models.graph_encoder_multiscale import HierarchicalGraphEncoder
from models.multiscale_classifier import MultiScaleGraphClassifier
from train.common import clone_state_dict_to_cpu
from train.token_utils import build_classifier_tokens
from train.contrastive import ProjectionHead, multi_scale_gene_contrastive_loss
from train.augmentation import augment_graph


def run_pretrain(train_dataset, out_dir, cfg: PretrainConfig, device: torch.device):
    """Run the main pretraining pipeline."""
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"[Pretrain] Samples: {len(train_dataset)}")
    logger.info(f"[Pretrain] Output dir: {out_dir}")

    # Single-process DataLoader for stability
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    encoder = HierarchicalGraphEncoder().to(device)
    proj_patch = ProjectionHead(cfg.HIDDEN_DIM, cfg.PROJ_DIM, cfg.PROJ_DIM).to(device)
    proj_region = ProjectionHead(cfg.HIDDEN_DIM, cfg.PROJ_DIM, cfg.PROJ_DIM).to(device)
    use_aux = getattr(cfg, "USE_AUX_CLASSIFICATION", True) and getattr(cfg, "AUX_CLS_WEIGHT", 0.0) > 0
    aux_classifier = MultiScaleGraphClassifier(token_dim=cfg.HIDDEN_DIM, num_classes=2).to(device) if use_aux else None
    aux_criterion = nn.CrossEntropyLoss(label_smoothing=getattr(cfg, "AUX_LABEL_SMOOTHING", 0.0)) if use_aux else None

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) +
        list(proj_patch.parameters()) +
        list(proj_region.parameters()) +
        (list(aux_classifier.parameters()) if use_aux else []),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    scaler = GradScaler('cuda', enabled=(device.type == "cuda"))
    accum_steps = max(1, int(getattr(cfg, "ACCUMULATION_STEPS", 1)))
    clip_params = (
        list(encoder.parameters())
        + list(proj_patch.parameters())
        + list(proj_region.parameters())
        + (list(aux_classifier.parameters()) if use_aux else [])
    )
    best_joint = float("inf")
    best_state = None

    for epoch in range(cfg.NUM_EPOCHS):
        encoder.train()
        proj_patch.train()
        proj_region.train()
        if use_aux:
            aux_classifier.train()

        total_loss = 0
        total_contrastive = 0
        total_aux = 0
        pbar = tqdm(loader, desc=f"[Pretrain] Epoch {epoch+1}/{cfg.NUM_EPOCHS}")

        optimizer.zero_grad(set_to_none=True)
        pending_micro = 0

        for batch in pbar:
            batch = batch.to(device)
            labels = batch.y.view(-1).to(device)

            # Build two augmented views with a shared node subset
            view1, shared_nodes = augment_graph(
                batch,
                drop_node_p=cfg.DROP_NODE_P,
                drop_edge_p=cfg.DROP_EDGE_P,
                mask_feat_p=cfg.MASK_FEAT_P,
                shared_nodes=None,
            )

            view2, _ = augment_graph(
                batch,
                drop_node_p=cfg.DROP_NODE_P,
                drop_edge_p=cfg.DROP_EDGE_P,
                mask_feat_p=cfg.MASK_FEAT_P,
                shared_nodes=shared_nodes,
            )

            # Skip empty graphs after augmentation
            if view1.x.size(0) == 0 or view2.x.size(0) == 0:
                continue

            with autocast('cuda', enabled=(device.type == "cuda")):
                o1 = encoder(view1)
                o2 = encoder(view2)

                # Defensive alignment: each graph should keep at least one node after
                # augmentation, but trim tokens/labels if counts ever diverge.
                n_graph = min(o1["patch_tokens"].size(0), o2["patch_tokens"].size(0), labels.size(0))
                if n_graph <= 0:
                    continue
                labels_eff = labels[:n_graph]
                p1_tok = o1["patch_tokens"][:n_graph]
                p2_tok = o2["patch_tokens"][:n_graph]
                r1_tok = o1["region_tokens"][:n_graph]
                r2_tok = o2["region_tokens"][:n_graph]

                p1 = proj_patch(p1_tok)
                p2 = proj_patch(p2_tok)
                r1 = proj_region(r1_tok)
                r2 = proj_region(r2_tok)

                contrastive_loss, _ = multi_scale_gene_contrastive_loss(
                    p1, p2, r1, r2,
                    labels=labels_eff,
                    temperature=cfg.TEMPERATURE,
                    weight_patch=0.5,
                    weight_region=0.5,
                )

                aux_loss = torch.zeros((), device=device)
                if use_aux:
                    tokens_v1 = build_classifier_tokens(p1_tok, r1_tok)
                    tokens_v2 = build_classifier_tokens(p2_tok, r2_tok)
                    logits_v1, _ = aux_classifier(tokens_v1)
                    logits_v2, _ = aux_classifier(tokens_v2)
                    aux_loss = 0.5 * (
                        aux_criterion(logits_v1, labels_eff) +
                        aux_criterion(logits_v2, labels_eff)
                    )
                loss = (contrastive_loss + getattr(cfg, "AUX_CLS_WEIGHT", 0.5) * aux_loss) / accum_steps

            scaler.scale(loss).backward()
            pending_micro += 1

            if pending_micro >= accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_micro = 0

            total_loss += loss.item() * accum_steps
            total_contrastive += contrastive_loss.item()
            total_aux += aux_loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        if pending_micro > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss = total_loss / max(len(loader), 1)
        epoch_contrastive = total_contrastive / max(len(loader), 1)
        epoch_aux = total_aux / max(len(loader), 1)
        logger.info(
            f"[Pretrain] Epoch {epoch+1} "
            f"joint={epoch_loss:.4f} contrastive={epoch_contrastive:.4f} aux_cls={epoch_aux:.4f}"
        )

        if epoch_loss < best_joint:
            best_joint = epoch_loss
            best_state = {
                "encoder": clone_state_dict_to_cpu(encoder),
                "proj_patch": clone_state_dict_to_cpu(proj_patch),
                "proj_region": clone_state_dict_to_cpu(proj_region),
                "aux_classifier": clone_state_dict_to_cpu(aux_classifier) if use_aux else None,
                "best_joint_loss": best_joint,
            }

        if (epoch + 1) % 20 == 0 or (epoch + 1) == cfg.NUM_EPOCHS:
            ckpt_path = os.path.join(out_dir, f"supcon_epoch_{epoch+1}.pt")
            torch.save({
                "encoder": encoder.state_dict(),
                "proj_patch": proj_patch.state_dict(),
                "proj_region": proj_region.state_dict(),
                "aux_classifier": aux_classifier.state_dict() if use_aux else None,
            }, ckpt_path)
            logger.info(f"[Pretrain] Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(out_dir, "encoder_final.pt")
    torch.save(encoder.state_dict(), final_path)
    if best_state is not None:
        torch.save(best_state, os.path.join(out_dir, "pretrain_joint_best.pt"))

    del encoder, proj_patch, proj_region, optimizer, scaler, loader
    if aux_classifier is not None:
        del aux_classifier
    del best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("[Pretrain] Finished; encoder_final.pt saved")

    return final_path
