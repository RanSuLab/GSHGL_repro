"""
Full training pipeline:
1. Multi-scale contrastive pretraining on all slides
2. Graph-level token extraction
3. Stratified cross-validation classification and result aggregation
"""

import numpy as np
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold

from configs.config import PathConfig, PretrainConfig, TrainConfig
from models.graph_encoder_multiscale import HierarchicalGraphEncoder
from models.multiscale_classifier import MultiScaleGraphClassifier
from train.common import (
    build_device,
    compute_binary_metrics,
    compute_scale_weight_stats,
    ensure_dir,
    load_cached_graph_dataset,
    run_path,
    save_json,
    summarize_fold_metrics,
)
from train.pretrain import run_pretrain
from train.token_utils import build_classifier_tokens


def _extract_all_tokens(encoder, data_list, device: torch.device):
    """Extract graph-level tokens for the full dataset."""
    encoder.eval()
    patch_list, region_list, label_list = [], [], []

    loader = PyGDataLoader(data_list, batch_size=1, shuffle=False)

    with torch.no_grad():
        for g in loader:
            g = g.to(device)
            out = encoder(g)
            patch_list.append(out["patch_tokens"].cpu())
            region_list.append(out["region_tokens"].cpu())
            label_list.append(int(g.y.item()))

    patch = torch.cat(patch_list, dim=0)
    region = torch.cat(region_list, dim=0)
    labels = torch.tensor(label_list, dtype=torch.long)

    if patch.shape[1] != region.shape[1]:
        proj = nn.Linear(region.shape[1], patch.shape[1])
        region = proj(region)

    tokens_all = build_classifier_tokens(patch, region)
    return tokens_all, labels


def run_training():
    """Run the full pipeline: pretrain -> token extraction -> cross-validation."""
    path_cfg = PathConfig()
    pre_cfg = PretrainConfig()
    train_cfg = TrainConfig()

    device = build_device(path_cfg.SELECTED_GPU)
    logger.info(f"[Train] Using device: {device}")

    data_list, _ = load_cached_graph_dataset(path_cfg.GRAPH_CACHE_DIR, path_cfg.GENE_NAME)
    logger.info(f"[Train] Loaded {len(data_list)} graph samples")

    encoder_path = Path(path_cfg.PRETRAIN_DIR) / "encoder_final.pt"
    if not encoder_path.exists():
        logger.warning("[Train] encoder_final.pt not found; starting pretraining")
        run_pretrain(data_list, path_cfg.PRETRAIN_DIR, pre_cfg, device)
    else:
        logger.info(f"[Train] Found pretrained encoder: {encoder_path}")

    encoder = HierarchicalGraphEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    tokens_all, labels_all = _extract_all_tokens(encoder, data_list, device)
    N, _, D = tokens_all.shape
    logger.info(f"[Train] Token shape: {tokens_all.shape}")

    dataset = TensorDataset(tokens_all, labels_all)
    root_out = path_cfg.RESULTS_DIR

    skf = StratifiedKFold(n_splits=train_cfg.N_SPLITS, shuffle=True, random_state=train_cfg.SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), labels_all.numpy()), start=1):
        logger.info("=" * 60)
        logger.info(f"[Train] Fold {fold}/{train_cfg.N_SPLITS}")

        fold_dir = f"fold_{fold}"
        ensure_dir(root_out, fold_dir)

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_cfg.BATCH_SIZE, shuffle=False)

        classifier = MultiScaleGraphClassifier(
            token_dim=D,
            num_classes=path_cfg.NUM_CLASSES,
        ).to(device)

        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=train_cfg.LEARNING_RATE,
            weight_decay=train_cfg.WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.LABEL_SMOOTHING)

        best_f1 = -1.0
        best_state = None
        patience = 0
        epoch_history = []

        for epoch in range(train_cfg.NUM_EPOCHS):
            classifier.train()
            losses = []

            for tokens, ys in train_loader:
                tokens, ys = tokens.to(device), ys.to(device)
                logits, info = classifier(tokens)
                loss = criterion(logits, ys)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())

            classifier.eval()
            preds, probs, labels = [], [], []
            scale_weights_all = []

            with torch.no_grad():
                for tokens, ys in val_loader:
                    tokens = tokens.to(device)
                    logits, info = classifier(tokens)
                    prob = torch.softmax(logits, dim=1)[:, 1]

                    preds.append(logits.argmax(dim=1).cpu().numpy())
                    probs.append(prob.cpu().numpy())
                    labels.append(ys.numpy())
                    scale_weights_all.append(info["scale_weights"].cpu().numpy())

            preds = np.concatenate(preds)
            probs = np.concatenate(probs)
            labels = np.concatenate(labels)
            metrics = compute_binary_metrics(labels, preds, probs)
            metrics.update(compute_scale_weight_stats(np.concatenate(scale_weights_all, axis=0)))

            logger.info(
                f"[Train][Fold {fold}] Epoch {epoch + 1} "
                f"loss={np.mean(losses):.4f} Acc={metrics['acc']:.4f} "
                f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} AUPR={metrics['aupr']:.4f}"
            )
            epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)) if losses else None,
                    "metrics": metrics,
                }
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {
                    "classifier": classifier.state_dict(),
                    "metrics": metrics,
                }
                patience = 0
            else:
                patience += 1
                if patience >= train_cfg.PATIENCE:
                    logger.info("[Train] Early stopping triggered")
                    break

        if best_state is None:
            raise RuntimeError(f"[Train] Fold {fold} produced no valid classifier checkpoint")

        torch.save(best_state["classifier"], run_path(root_out, f"{fold_dir}/best_classifier.pt"))
        save_json(best_state["metrics"], run_path(root_out, f"{fold_dir}/fold_summary.json"))
        save_json(epoch_history, run_path(root_out, f"{fold_dir}/epoch_history.json"))
        fold_metrics.append(best_state["metrics"])

    summary = summarize_fold_metrics(fold_metrics)
    summary_path = run_path(root_out, "summary.json")
    save_json(summary, summary_path)

    logger.info("=" * 60)
    logger.info("[Train] Done")
    logger.info(f"[Train] Results saved to: {summary_path}")
