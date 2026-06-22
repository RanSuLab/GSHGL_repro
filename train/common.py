"""Shared training utilities: device, metrics, cache loading, and result I/O."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch_geometric.data import Data


def build_device(selected_gpu: int) -> torch.device:
    """Build the training device from a GPU index."""
    return torch.device(f"cuda:{selected_gpu}" if torch.cuda.is_available() else "cpu")


def clone_state_dict_to_cpu(module_or_state_dict):
    """Clone state_dict to CPU to avoid OOM from deepcopy retaining full GPU weights."""
    if hasattr(module_or_state_dict, "state_dict"):
        sd = module_or_state_dict.state_dict()
    else:
        sd = module_or_state_dict
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def load_cached_graph_dataset(cache_dir: str, gene_name: str):
    """Load the graph cache produced by the preprocessing stage."""
    d = Path(cache_dir)
    torch.serialization.add_safe_globals([Data])

    hybrid = d / f"cache_graphs_hybrid_{gene_name}.pt"
    if hybrid.is_file():
        cache_file = hybrid
    else:
        matches = sorted(d.glob(f"cache_graphs_*_{gene_name}.pt"))
        if not matches:
            existing = sorted(p.name for p in d.glob("*.pt")) if d.is_dir() else []
            raise FileNotFoundError(
                f"Graph cache not found: gene={gene_name}, dir={d}. "
                f"Expected cache_graphs_<tag>_{gene_name}.pt; .pt files in dir: {existing}"
            )
        cache_file = matches[0]

    return torch.load(cache_file, weights_only=False)


def compute_binary_metrics(labels, preds, probs):
    """Compute standard binary classification metrics."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    probs = np.asarray(probs)

    metrics = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except Exception:
        metrics["auc"] = float("nan")

    try:
        metrics["aupr"] = float(average_precision_score(labels, probs))
    except Exception:
        metrics["aupr"] = float("nan")

    return metrics


def summarize_fold_metrics(fold_metrics):
    """Summarize per-fold metrics into mean and standard deviation."""
    keys = fold_metrics[0].keys()
    summary_mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    summary_std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {"mean": summary_mean, "std": summary_std}


def compute_scale_weight_stats(scale_weights):
    """Compute mean and std of per-scale attention weights."""
    scale_weights = np.asarray(scale_weights)
    stats = {}
    if scale_weights.size == 0:
        return stats
    if scale_weights.ndim == 1:
        scale_weights = scale_weights[:, None]
    names = ["patch", "region"]
    stats[f"{names[0]}_scale_mean"] = float(scale_weights[:, 0].mean())
    stats[f"{names[0]}_scale_std"] = float(scale_weights[:, 0].std())
    if scale_weights.shape[1] > 1:
        stats[f"{names[1]}_scale_mean"] = float(scale_weights[:, 1].mean())
        stats[f"{names[1]}_scale_std"] = float(scale_weights[:, 1].std())
    return stats


def ensure_dir(root: str | Path, relative: str | Path) -> Path:
    """Create and return a subdirectory under the experiment root."""
    target = Path(root) / relative
    target.mkdir(parents=True, exist_ok=True)
    return target


def run_path(root: str | Path, relative: str | Path) -> Path:
    """Return an absolute path under the experiment root."""
    return Path(root) / relative


def save_json(payload, path):
    """Save a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
