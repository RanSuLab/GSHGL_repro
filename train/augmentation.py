"""
Graph augmentation for constructing dual views in supervised contrastive pretraining.
"""

import torch
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Data


def mask_features(x, p=0.1):
    if p <= 0:
        return x
    mask = torch.rand_like(x) < p
    return x * (~mask)


def _apply_node_subset(data: Data, nodes_to_keep: torch.Tensor):
    data = data.clone()
    num_nodes = data.num_nodes

    mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=data.x.device)
    mapping[nodes_to_keep] = torch.arange(len(nodes_to_keep), device=data.x.device)

    data.x = data.x[nodes_to_keep]
    if hasattr(data, "pos") and data.pos is not None:
        data.pos = data.pos[nodes_to_keep]
    if hasattr(data, "batch") and data.batch is not None:
        data.batch = data.batch[nodes_to_keep]

    row, col = data.edge_index
    mask = (mapping[row] >= 0) & (mapping[col] >= 0)
    data.edge_index = torch.stack([mapping[row[mask]], mapping[col[mask]]], dim=0)
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = data.edge_attr[mask]

    return data


def augment_graph(
    data: Data,
    drop_node_p=0.1,
    drop_edge_p=0.1,
    mask_feat_p=0.1,
    shared_nodes=None,
):
    """
    Build an augmented view. Both views share the same node subset so contrastive
    learning remains alignable. Returns the augmented graph and the shared node indices.
    """
    data = data.clone()
    num_nodes = data.num_nodes

    if shared_nodes is None:
        if drop_node_p > 0 and num_nodes > 1:
            batch = getattr(data, "batch", None)
            if batch is None:
                num_to_keep = max(1, int(num_nodes * (1 - drop_node_p)))
                perm = torch.randperm(num_nodes, device=data.x.device)
                shared_nodes = perm[:num_to_keep]
            else:
                keep_parts = []
                graph_ids = torch.unique(batch, sorted=True)
                for gid in graph_ids:
                    idx = torch.nonzero(batch == gid, as_tuple=False).view(-1)
                    n = idx.numel()
                    if n == 0:
                        continue
                    n_keep = max(1, int(n * (1 - drop_node_p)))
                    perm_local = torch.randperm(n, device=data.x.device)
                    keep_parts.append(idx[perm_local[:n_keep]])
                shared_nodes = (
                    torch.cat(keep_parts, dim=0)
                    if keep_parts
                    else torch.arange(num_nodes, device=data.x.device)
                )
        else:
            shared_nodes = torch.arange(num_nodes, device=data.x.device)

    data = _apply_node_subset(data, shared_nodes)

    if drop_edge_p > 0:
        data.edge_index, edge_mask = dropout_edge(data.edge_index, p=drop_edge_p)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr[edge_mask]

    if mask_feat_p > 0:
        data.x = mask_features(data.x, p=mask_feat_p)

    return data, shared_nodes
