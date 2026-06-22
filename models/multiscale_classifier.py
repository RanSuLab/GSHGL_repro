"""
Multi-scale classifier with explicit cross-scale interaction.

Compared with simple scale weighting, this version:
- Adds learnable scale embeddings
- Uses a lightweight Transformer encoder for patch/region interaction
- Keeps scale weights for interpretability
"""

import torch
import torch.nn as nn
from configs.config import PretrainConfig, TrainConfig


class MultiScaleGraphClassifier(nn.Module):
    def __init__(self, token_dim=None, num_classes=None):
        super().__init__()

        pre_cfg = PretrainConfig()
        train_cfg = TrainConfig()

        self.token_dim = token_dim if token_dim is not None else pre_cfg.HIDDEN_DIM
        self.num_classes = num_classes if num_classes is not None else 2
        self.ff_dim = getattr(train_cfg, "TRANSFORMER_FF_DIM", max(self.token_dim * 2, 512))
        self.dropout = getattr(train_cfg, "FUSION_DROPOUT", 0.2)
        self.num_heads = getattr(train_cfg, "FUSION_NUM_HEADS", 4)
        self.num_layers = getattr(train_cfg, "FUSION_NUM_LAYERS", 2)
        self.fusion_type = getattr(train_cfg, "FUSION_TYPE", "cross_scale_transformer")

        self.input_norm = nn.LayerNorm(self.token_dim)
        self.scale_embeddings = nn.Parameter(torch.randn(8, self.token_dim) * 0.02)

        if self.fusion_type == "cross_scale_transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.token_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.cross_scale_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        elif self.fusion_type == "scale_attention":
            self.cross_scale_encoder = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        self.scale_attn = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim, self.ff_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ff_dim // 2, self.num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _prepare_tokens(self, tokens):
        if isinstance(tokens, (list, tuple)):
            tokens_list = []
            for t in tokens:
                if t.dim() == 2:
                    tokens_list.append(t.unsqueeze(1))
                elif t.dim() == 3:
                    tokens_list.append(t)
                else:
                    raise ValueError(f"Unsupported token tensor shape: {t.shape}")
            tokens = torch.cat(tokens_list, dim=1)

        if not isinstance(tokens, torch.Tensor):
            raise ValueError("tokens must be a Tensor or list/tuple[Tensor]")
        if tokens.dim() != 3:
            raise ValueError("tokens must have shape [B, S, D]")
        if tokens.size(-1) != self.token_dim:
            raise ValueError(f"Input token dim {tokens.size(-1)} != model token_dim {self.token_dim}")
        return tokens

    def forward(self, tokens):
        tokens = self._prepare_tokens(tokens)
        _, num_scales, _ = tokens.shape

        if num_scales > self.scale_embeddings.size(0):
            raise ValueError(f"Model supports at most {self.scale_embeddings.size(0)} scale tokens")

        tokens = self.input_norm(tokens)
        if num_scales > 1:
            tokens = tokens + self.scale_embeddings[:num_scales].unsqueeze(0)

        tokens_cross_scale = self.cross_scale_encoder(tokens)
        attn_logits = self.scale_attn(tokens_cross_scale)
        scale_weights = torch.softmax(attn_logits, dim=1)
        fused_repr = (tokens_cross_scale * scale_weights).sum(dim=1)
        logits = self.classifier(fused_repr)

        info = {
            "scale_weights": scale_weights.squeeze(-1),
            "tokens_refined": tokens_cross_scale,
            "fused_repr": fused_repr,
        }
        return logits, info


if __name__ == "__main__":
    B, S, D = 4, 2, 512
    dummy = torch.randn(B, S, D)
    model = MultiScaleGraphClassifier(token_dim=D, num_classes=2)
    logits, info = model(dummy)
    print("logits shape:", logits.shape)
    print("scale weights shape:", info["scale_weights"].shape)
