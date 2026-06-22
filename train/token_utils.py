"""Token normalization and classifier input construction."""

import torch
import torch.nn.functional as F

from configs.config import TrainConfig


def normalize_encoder_tokens(out_dict):
    """Layer-normalize patch and region tokens from the encoder output."""
    patch_tokens = F.layer_norm(out_dict["patch_tokens"], out_dict["patch_tokens"].shape[-1:])
    region_tokens = F.layer_norm(out_dict["region_tokens"], out_dict["region_tokens"].shape[-1:])
    return patch_tokens, region_tokens


def build_classifier_tokens(
    patch_tokens,
    region_tokens,
    token_scheme=None,
):
    """Build the token sequence fed into the classifier."""
    train_cfg = TrainConfig()
    token_scheme = token_scheme or getattr(train_cfg, "TOKEN_SCHEME", "patch_region")

    if token_scheme == "patch_only":
        return patch_tokens.unsqueeze(1)
    if token_scheme == "region_only":
        return region_tokens.unsqueeze(1)
    if token_scheme == "patch_region":
        return torch.stack([patch_tokens, region_tokens], dim=1)
    raise ValueError(f"Unsupported token scheme: {token_scheme}")
