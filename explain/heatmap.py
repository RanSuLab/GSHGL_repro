"""
Gradient-based node heatmap for WSI graphs.

Visualization enhancements for clearer hot regions:
- Percentile clipping to reduce outlier dominance
- Highlight only the top-k% nodes; render the rest as light gray background
- Optional debug output of score quantiles

Inputs:
- Single graph ``.pt`` (saved by ``preprocess.py``)
- Encoder weights ``best_encoder.pt`` or ``encoder_final.pt``
- Classifier weights ``best_classifier.pt``

Output:
- Scatter heatmap ``png`` based on node importance
"""
import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parents[1]
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

import os
import argparse
import numpy as np  # Import before torch to avoid MKL_THREADING_LAYER vs libgomp conflicts
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from models.graph_encoder_multiscale import HierarchicalGraphEncoder
from models.multiscale_classifier import MultiScaleGraphClassifier
from train.token_utils import build_classifier_tokens, normalize_encoder_tokens
from configs.config import PathConfig, TrainConfig


def load_graph(pt_path: str, device: torch.device) -> Data:
    torch.serialization.add_safe_globals([Data])
    data = torch.load(pt_path, weights_only=False)
    if not hasattr(data, "x") or not hasattr(data, "pos"):
        raise ValueError("Graph file must contain data.x and data.pos (node features and coordinates)")
    return data.to(device)


def load_models(encoder_ckpt: str, classifier_ckpt: str, device: torch.device):
    path_cfg = PathConfig()
    train_cfg = TrainConfig()

    encoder = HierarchicalGraphEncoder().to(device)
    enc_obj = torch.load(encoder_ckpt, map_location=device)
    # Support plain state_dict or {"encoder": state_dict, ...}
    if isinstance(enc_obj, dict) and "encoder" in enc_obj and isinstance(enc_obj["encoder"], dict):
        enc_state = enc_obj["encoder"]
    else:
        enc_state = enc_obj

    if isinstance(enc_state, dict) and any(k.startswith(("cross_scale_encoder", "scale_attn", "classifier")) for k in enc_state.keys()):
        raise RuntimeError(
            "--encoder appears to be a classifier checkpoint (contains cross_scale_encoder/scale_attn/classifier). "
            "Check that the paths are not swapped."
        )

    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()

    classifier = MultiScaleGraphClassifier(
        token_dim=train_cfg.HIDDEN_DIM,
        num_classes=path_cfg.NUM_CLASSES
    ).to(device)

    clf_obj = torch.load(classifier_ckpt, map_location=device)
    if isinstance(clf_obj, dict) and "classifier" in clf_obj and isinstance(clf_obj["classifier"], dict):
        clf_state = clf_obj["classifier"]
    else:
        clf_state = clf_obj

    if isinstance(clf_state, dict) and any(k.startswith(("gat_patch_", "gat_region_", "proj_patch", "proj_region")) for k in clf_state.keys()):
        raise RuntimeError(
            "--classifier appears to be an encoder checkpoint (contains gat_patch_/gat_region_). "
            "Check that the paths are not swapped."
        )

    classifier.load_state_dict(clf_state, strict=True)
    classifier.eval()

    return encoder, classifier


def grad_importance(encoder, classifier, data: Data, target_class: int, method: str = "grad_x_input"):
    """
    Gradient attribution on node features.

    Returns:
    - ``scores``: node importance in ``[N]``, normalized to ``[0, 1]``
    - ``pred_prob``: predicted probability for the target class
    - ``scale_weights``: scale weights if available
    """
    data = data.clone()
    data.x = data.x.detach().requires_grad_(True)

    out = encoder(data)
    patch_tokens, region_tokens = normalize_encoder_tokens(out)
    tokens = build_classifier_tokens(patch_tokens, region_tokens)

    logits, info = classifier(tokens)
    prob = torch.softmax(logits, dim=1)[0, target_class]
    target_logit = logits[0, target_class]
    target_logit.backward()

    grad = data.x.grad
    x = data.x.detach()

    if method == "grad_norm":
        scores = torch.norm(grad, p=2, dim=1)
    elif method == "grad_x_input":
        scores = torch.norm(grad * x, p=2, dim=1)
    else:
        raise ValueError("method must be one of: grad_norm, grad_x_input")

    scores = scores.detach().float().cpu().numpy()
    # Base normalization; plotting applies further percentile clip and top-k highlight
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    scale_w = info.get("scale_weights", None)
    if scale_w is not None:
        scale_w = scale_w.detach().cpu().numpy()[0]

    return scores, float(prob.detach().cpu().item()), scale_w


def plot_heatmap_topk(
    data: Data,
    scores: np.ndarray,
    out_png: str,
    title: str = "",
    top_frac: float = 0.05,
    clip_percentile: float = 99.0,
    background_alpha: float = 0.25,
    point_size_bg: float = 6.0,
    point_size_fg: float = 16.0,
):
    """
    Improve hot-region visibility via clipping and highlighting.

    - Clip to ``pXX`` percentile (default ``p99``)
    - Highlight only the top-k% points; render the rest as gray background
    """
    pos = data.pos.detach().cpu().numpy()
    if pos.shape[1] < 2:
        xs = np.arange(len(scores))
        ys = np.zeros_like(xs)
    else:
        xs, ys = pos[:, 0], pos[:, 1]

    s = scores.copy()

    vmax = np.percentile(s, clip_percentile)
    s = np.clip(s, 0.0, vmax)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)

    N = len(s)
    k = max(1, int(N * top_frac))
    idx = np.argsort(s)[-k:]

    plt.figure(figsize=(8, 6))

    plt.scatter(xs, ys, c="lightgray", s=point_size_bg, alpha=background_alpha, linewidths=0)

    sc = plt.scatter(xs[idx], ys[idx], c=s[idx], s=point_size_fg, alpha=0.95, linewidths=0)

    plt.gca().invert_yaxis()
    plt.colorbar(sc, label=f"Importance (clip p{clip_percentile}, top {int(top_frac*100)}%)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to a single slide graph .pt")
    ap.add_argument("--encoder", required=True, help="Encoder checkpoint, e.g. best_encoder.pt or encoder_final.pt")
    ap.add_argument("--classifier", required=True, help="Classifier checkpoint, e.g. best_classifier.pt")
    ap.add_argument("--out", default="heatmap.png", help="Output image path")
    ap.add_argument("--cls", type=int, default=1, help="Target class index, e.g. 1 for mutation")
    ap.add_argument("--method", choices=["grad_x_input", "grad_norm"], default="grad_x_input")
    ap.add_argument("--device", default="cuda:0", help="Device, e.g. cpu, cuda, or cuda:0")

    ap.add_argument("--top_frac", type=float, default=0.05, help="Fraction of top nodes to highlight, e.g. 0.05")
    ap.add_argument("--clip_p", type=float, default=99.0, help="Percentile clip, e.g. 99")
    ap.add_argument("--debug", action="store_true", help="Print score quantiles for debugging")

    args = ap.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    data = load_graph(args.pt, device)
    encoder, classifier = load_models(args.encoder, args.classifier, device)

    scores, prob, scale_w = grad_importance(encoder, classifier, data, args.cls, args.method)

    if args.debug:
        pcts = [0, 50, 90, 95, 99, 99.5, 99.9, 100]
        print("Score quantiles:", {p: float(np.percentile(scores, p)) for p in pcts})

    title = f"Heatmap | target_cls={args.cls} prob={prob:.3f}"
    if scale_w is not None and len(scale_w) >= 2:
        title += f" | scale_w(patch,region)=({scale_w[0]:.2f},{scale_w[1]:.2f})"

    plot_heatmap_topk(
        data=data,
        scores=scores,
        out_png=args.out,
        title=title,
        top_frac=args.top_frac,
        clip_percentile=args.clip_p,
    )

    print(f"[Done] Saved to: {args.out}")


if __name__ == "__main__":
    main()
