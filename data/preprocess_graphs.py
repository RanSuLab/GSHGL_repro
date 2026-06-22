"""
Build hybrid graph data (.pt) from {gene_name}_labels.pkl.

Graph structure combines:
- Spatial neighborhood relations
- Feature similarity relations

Edge attributes include:
- dx, dy
- spatial distance
- feature cosine similarity
"""

import os
from pathlib import Path
import joblib
import numpy as np
import torch
from torch_geometric.data import Data
from loguru import logger
from tqdm import tqdm

from configs.config import PathConfig, PreprocessConfig


def compute_adaptive_k(N, base_n_neighbors):
    """Adaptive K based on graph size."""
    if N <= 3:
        return max(1, N - 1)
    k = max(3, int(max(3, N * 0.08)))
    return min(k, base_n_neighbors)


@torch.no_grad()
def _normalize_coords(coords):
    coords_min = coords.min(dim=0, keepdim=True).values
    coords_max = coords.max(dim=0, keepdim=True).values
    return (coords - coords_min) / (coords_max - coords_min + 1e-8)


@torch.no_grad()
def _normalize_features(features):
    return torch.nn.functional.normalize(features, dim=1, eps=1e-8)


@torch.no_grad()
def build_hybrid_graph(
    coords_np,
    feat_np,
    spatial_neighbors,
    feature_neighbors,
    adaptive=True,
    normalize_coords=True,
    normalize_features=True,
    device="cpu",
):
    """Build a spatial + feature-similarity hybrid graph; return edge_index / edge_attr."""
    coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
    feat = torch.tensor(feat_np, dtype=torch.float32, device=device)
    N = coords.size(0)

    if N < 2:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )

    spatial_k = compute_adaptive_k(N, spatial_neighbors) if adaptive else spatial_neighbors
    feature_k = min(max(0, feature_neighbors), max(0, N - 1))

    if normalize_coords:
        coords = _normalize_coords(coords)
    if normalize_features:
        feat = _normalize_features(feat)

    # Spatial neighbors
    spatial_dist = torch.cdist(coords, coords, p=2)
    spatial_dist.fill_diagonal_(float("inf"))
    spatial_k = min(spatial_k, max(1, N - 1))
    _, spatial_idx = torch.topk(spatial_dist, k=spatial_k, dim=1, largest=False)

    # Feature neighbors by cosine similarity
    feature_sim = torch.matmul(feat, feat.T)
    feature_sim.fill_diagonal_(-float("inf"))
    feature_idx = None
    if feature_k > 0:
        _, feature_idx = torch.topk(feature_sim, k=feature_k, dim=1, largest=True)

    adjacency = torch.zeros((N, N), dtype=torch.bool, device=device)
    row = torch.arange(N, device=device).view(-1, 1)
    adjacency[row, spatial_idx] = True
    if feature_idx is not None:
        adjacency[row, feature_idx] = True
    adjacency.fill_diagonal_(False)
    adjacency = adjacency | adjacency.T

    row_idx, col_idx = adjacency.nonzero(as_tuple=True)
    edge_index = torch.stack([row_idx, col_idx], dim=0)

    deltas = coords[col_idx] - coords[row_idx]
    dist_attr = torch.norm(deltas, p=2, dim=1, keepdim=True)
    sim_attr = feature_sim[row_idx, col_idx].unsqueeze(1).clamp(min=-1.0, max=1.0)
    edge_attr = torch.cat([deltas, dist_attr, sim_attr], dim=1)

    return edge_index.cpu(), edge_attr.cpu()


def main():
    path_cfg = PathConfig()
    preprocess_cfg = PreprocessConfig()

    slide_dir = Path(path_cfg.GRAPH_SLIDE_DIR)
    cache_dir = Path(path_cfg.GRAPH_CACHE_DIR)

    os.makedirs(slide_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    pkl_path = Path(path_cfg.PKL_PATH)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Input file not found: {pkl_path}")

    logger.info(f"Preprocessing {pkl_path.name}")
    logger.info(f"Graph output dir: {slide_dir}")

    prepared = joblib.load(str(pkl_path))
    names_list = prepared['names_list']
    coords_list = prepared['coords_list']
    features_list = prepared['features_list']
    labels_list = prepared['labels_list']

    n_samples = len(features_list)
    logger.info(f"Loaded {n_samples} WSI samples")

    device = torch.device(f"cuda:{path_cfg.SELECTED_GPU}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    saved, failed = 0, 0
    data_list = []
    labels_out = []

    pbar = tqdm(range(n_samples), desc="Building graphs")
    for idx in pbar:
        try:
            feat = np.asarray(features_list[idx], dtype=np.float32)
            coord = np.asarray(coords_list[idx], dtype=np.float32)
            label = int(labels_list[idx])
            name = names_list[idx]

            # Drop invalid features or coordinates
            valid = np.isfinite(feat).all(axis=1) & np.isfinite(coord).all(axis=1)
            feat = feat[valid]
            coord = coord[valid]

            if feat.shape[0] < preprocess_cfg.MIN_PATCHES:
                logger.warning(f"Skipping {name}: insufficient patches ({feat.shape[0]})")
                failed += 1
                continue

            edge_index, edge_attr = build_hybrid_graph(
                coord,
                feat,
                spatial_neighbors=preprocess_cfg.MAX_NEIGHBORS,
                feature_neighbors=preprocess_cfg.FEATURE_K if preprocess_cfg.USE_HYBRID_GRAPH else 0,
                adaptive=preprocess_cfg.ADAPTIVE_K,
                normalize_coords=preprocess_cfg.NORMALIZE_COORDS,
                normalize_features=preprocess_cfg.NORMALIZE_FEATURES,
                device=device,
            )

            data = Data(
                x=torch.from_numpy(feat),
                pos=torch.from_numpy(coord),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(label, dtype=torch.long)
            )

            # Save per-slide graph
            fname = slide_dir / f"{idx:04d}_{name}_{label}.pt"
            torch.save(data, fname)

            data_list.append(data)
            labels_out.append(label)

            saved += 1
        except Exception as e:
            logger.error(f"Error at index {idx}: {e}")
            failed += 1

    logger.info(f"Graph construction done: {saved}/{n_samples} succeeded, {failed} failed")

    # Save full-dataset cache
    cache_path = Path(path_cfg.GRAPH_CACHE_FILE)
    torch.save((data_list, torch.tensor(labels_out)), cache_path)
    logger.info("Cache saved: {} | absolute path {}", cache_path, cache_path.resolve())


if __name__ == "__main__":
    main()
