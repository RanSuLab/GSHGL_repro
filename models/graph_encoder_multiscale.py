"""
Edge-aware hierarchical graph encoder.

Uses hybrid edges from spatial and feature neighbors, with edge attributes
describing relative position and feature similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, TopKPooling, global_max_pool, global_mean_pool

from configs.config import PretrainConfig


class GATEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.5, edge_dim=4):
        super().__init__()
        self.gat = GATConv(
            in_dim,
            out_dim // heads,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        res = self.proj(x)
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.bn(x)
        x = self.dropout(x + res)
        return x


class HierarchicalGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = PretrainConfig()

        in_dim = cfg.INPUT_DIM
        hidden_dim_1 = cfg.HIDDEN_DIM
        hidden_dim_2 = cfg.HIDDEN_DIM
        heads = cfg.GAT_HEADS
        dropout = cfg.DROPOUT
        edge_dim = getattr(cfg, "EDGE_DIM", 4)

        self.gat_patch_1 = GATEncoder(in_dim, hidden_dim_1, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.gat_patch_2 = GATEncoder(hidden_dim_1, hidden_dim_1, heads=heads, dropout=dropout, edge_dim=edge_dim)

        self.pool_patch = TopKPooling(hidden_dim_1, ratio=0.25)
        self.gat_region_1 = GATEncoder(hidden_dim_1, hidden_dim_2, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.gat_region_2 = GATEncoder(hidden_dim_2, hidden_dim_2, heads=heads, dropout=dropout, edge_dim=edge_dim)

        self.proj_patch = nn.Sequential(
            nn.Linear(hidden_dim_1 * 2, hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.proj_region = nn.Sequential(
            nn.Linear(hidden_dim_2 * 2, hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self._grad_checkpointing = bool(getattr(cfg, "GRADIENT_CHECKPOINTING", False))

    def _encode_patch_gats(self, x, edge_index, edge_attr):
        x = self.gat_patch_1(x, edge_index, edge_attr=edge_attr)
        x = self.gat_patch_2(x, edge_index, edge_attr=edge_attr)
        return x

    def _encode_region_gats(self, x_pooled, edge_index_pooled, edge_attr_pooled):
        x = self.gat_region_1(x_pooled, edge_index_pooled, edge_attr=edge_attr_pooled)
        x = self.gat_region_2(x, edge_index_pooled, edge_attr=edge_attr_pooled)
        return x

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        use_ckpt = self._grad_checkpointing and self.training
        if use_ckpt:
            x = checkpoint(
                self._encode_patch_gats,
                x,
                edge_index,
                edge_attr,
                use_reentrant=False,
            )
        else:
            x = self._encode_patch_gats(x, edge_index, edge_attr)

        x_patch_mean = global_mean_pool(x, batch)
        x_patch_max = global_max_pool(x, batch)
        patch_token = self.proj_patch(torch.cat([x_patch_mean, x_patch_max], dim=-1))

        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, _, _ = self.pool_patch(
            x,
            edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )
        if use_ckpt:
            x_pooled = checkpoint(
                self._encode_region_gats,
                x_pooled,
                edge_index_pooled,
                edge_attr_pooled,
                use_reentrant=False,
            )
        else:
            x_pooled = self._encode_region_gats(x_pooled, edge_index_pooled, edge_attr_pooled)

        x_region_mean = global_mean_pool(x_pooled, batch_pooled)
        x_region_max = global_max_pool(x_pooled, batch_pooled)
        region_token = self.proj_region(torch.cat([x_region_mean, x_region_max], dim=-1))

        return {"patch_tokens": patch_token, "region_tokens": region_token}
