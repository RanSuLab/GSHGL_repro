"""
Multi-scale gene-supervised contrastive loss and projection heads.
"""

import torch
import torch.nn.functional as F
from configs.config import PretrainConfig
from typing import Tuple, Optional

cfg = PretrainConfig()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = None,
) -> torch.Tensor:
    """Supervised contrastive loss with robust handling of edge cases."""
    if temperature is None:
        temperature = cfg.TEMPERATURE

    device = features.device
    if features.size(0) <= 1:
        return features.sum() * 0.0

    N = features.size(0)
    labels = labels.view(-1).to(device)
    if labels.size(0) != N:
        labels = labels.repeat((N // labels.size(0)) + 1)[:N]

    features = F.normalize(features, dim=1, eps=1e-6)

    try:
        sim = torch.matmul(features, features.T) / temperature
    except Exception:
        return features.sum() * 0.0

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max.detach()
    exp_sim = torch.exp(sim)

    try:
        mask = torch.eq(labels.view(N, 1), labels.view(1, N)).float()
    except Exception:
        return features.sum() * 0.0

    if mask.size(0) != N or mask.size(1) != N:
        new_mask = torch.zeros((N, N), device=device)
        h = min(mask.size(0), N)
        w = min(mask.size(1), N)
        new_mask[:h, :w] = mask[:h, :w]
        mask = new_mask

    mask = mask - torch.eye(N, device=device)

    denom = exp_sim.sum(dim=1) - torch.exp(sim.diag())
    pos_exp = (exp_sim * mask).sum(dim=1)

    valid = (pos_exp > 0).float()
    if valid.sum() < 1:
        return features.sum() * 0.0

    eps = 1e-12
    loss_i = -torch.log((pos_exp + eps) / (denom + eps)) * valid
    return loss_i.sum() / valid.sum()


def multi_scale_gene_contrastive_loss(
    patch_proj_1: torch.Tensor,
    patch_proj_2: torch.Tensor,
    region_proj_1: Optional[torch.Tensor],
    region_proj_2: Optional[torch.Tensor],
    labels: torch.Tensor,
    temperature: Optional[float] = None,
    weight_patch: float = 0.5,
    weight_region: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """Joint contrastive loss at patch and region scales."""
    if temperature is None:
        temperature = cfg.TEMPERATURE

    device = patch_proj_1.device
    patch_feat = torch.cat([patch_proj_1, patch_proj_2], dim=0)
    patch_labels = torch.cat([labels, labels], dim=0)
    loss_patch = supervised_contrastive_loss(patch_feat, patch_labels, temperature)

    if region_proj_1 is not None and region_proj_2 is not None:
        region_feat = torch.cat([region_proj_1, region_proj_2], dim=0)
        region_labels = torch.cat([labels, labels], dim=0)
        loss_region = supervised_contrastive_loss(region_feat, region_labels, temperature)
    else:
        loss_region = torch.tensor(0.0, device=device)

    total = weight_patch * loss_patch + weight_region * loss_region
    return total, {
        "loss_total": float(total.detach().cpu().item()),
        "loss_patch": float(loss_patch.detach().cpu().item()),
        "loss_region": float(loss_region.detach().cpu().item()),
    }


class ProjectionHead(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
