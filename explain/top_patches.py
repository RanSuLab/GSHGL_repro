"""
Interpretability script for top patches on the original WSI.
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
from PIL import Image, ImageDraw, ImageFont
import openslide
from torch_geometric.data import Data

from models.graph_encoder_multiscale import HierarchicalGraphEncoder
from models.multiscale_classifier import MultiScaleGraphClassifier
from train.token_utils import build_classifier_tokens, normalize_encoder_tokens
from configs.config import PathConfig, TrainConfig


def load_graph(pt_path: str, device):
    torch.serialization.add_safe_globals([Data])
    data = torch.load(pt_path, weights_only=False)
    if not hasattr(data, "x") or not hasattr(data, "pos"):
        raise ValueError("Graph file must contain data.x and data.pos")
    return data.to(device)


def load_models(encoder_ckpt: str, classifier_ckpt: str, device):
    path_cfg = PathConfig()
    train_cfg = TrainConfig()

    encoder = HierarchicalGraphEncoder().to(device)
    enc_obj = torch.load(encoder_ckpt, map_location=device)
    enc_state = enc_obj["encoder"] if isinstance(enc_obj, dict) and "encoder" in enc_obj else enc_obj
    if isinstance(enc_state, dict) and any(k.startswith(("cross_scale_encoder", "scale_attn", "classifier")) for k in enc_state.keys()):
        raise RuntimeError("--encoder appears to be a classifier ckpt (contains cross_scale_encoder/scale_attn/classifier)")
    encoder.load_state_dict(enc_state, strict=True)
    encoder.eval()

    classifier = MultiScaleGraphClassifier(
        token_dim=train_cfg.HIDDEN_DIM,
        num_classes=path_cfg.NUM_CLASSES
    ).to(device)
    clf_obj = torch.load(classifier_ckpt, map_location=device)
    clf_state = clf_obj["classifier"] if isinstance(clf_obj, dict) and "classifier" in clf_obj else clf_obj
    if isinstance(clf_state, dict) and any(k.startswith(("gat_patch_", "gat_region_", "proj_patch", "proj_region")) for k in clf_state.keys()):
        raise RuntimeError("--classifier appears to be an encoder ckpt (contains gat_patch_/gat_region_)")
    classifier.load_state_dict(clf_state, strict=True)
    classifier.eval()

    return encoder, classifier


def grad_importance(encoder, classifier, data: Data, target_class: int, method: str = "grad_x_input"):
    data = data.clone()
    data.x = data.x.detach().requires_grad_(True)

    out = encoder(data)
    patch_tokens, region_tokens = normalize_encoder_tokens(out)
    tokens = build_classifier_tokens(patch_tokens, region_tokens)

    logits, info = classifier(tokens)
    prob = torch.softmax(logits, dim=1)[0, target_class]
    logits[0, target_class].backward()

    grad = data.x.grad
    x = data.x.detach()

    if method == "grad_norm":
        s = torch.norm(grad, p=2, dim=1)
    else:  # grad_x_input method
        s = torch.norm(grad * x, p=2, dim=1)

    s_raw = s.detach().cpu().numpy().astype(np.float32)

    scale_w = info.get("scale_weights", None)
    if scale_w is not None:
        scale_w = scale_w.detach().cpu().numpy()[0]

    return s_raw, float(prob.detach().cpu().item()), scale_w


def format_attribution_scores(
    all_raw: np.ndarray,
    values: list[float] | np.ndarray,
    mode: str = "robust",
) -> list[float]:
    """
    Map raw gradient attribution to publication-friendly scores (avoids per-slide min-max making top-1 always 1.00).
    robust: slide-wide 2-98 percentile + power compression to ~[0.30, 0.92]
    percentile: slide-wide empirical percentile [0, 1]
    topk_spread: spread only within top-k to [0.42, 0.88]
    """
    all_raw = np.asarray(all_raw, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return []

    if mode == "percentile":
        return [float(np.clip((all_raw <= v).mean(), 0.0, 1.0)) for v in vals]

    if mode == "topk_spread":
        if vals.size == 1:
            return [0.72]
        rel = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        return [float(0.42 + 0.46 * r) for r in rel]

    if mode == "paper":
        robust = format_attribution_scores(all_raw, values, mode="robust")
        if np.ptp(robust) >= 0.06:
            return robust
        return format_attribution_scores(all_raw, values, mode="topk_spread")

    # robust
    lo, hi = np.percentile(all_raw, [2.0, 98.0])
    hi = max(float(hi), float(lo) + 1e-8)
    out: list[float] = []
    for v in vals:
        t = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        out.append(0.30 + 0.62 * (t ** 0.72))
    return out


def read_patch(slide: openslide.OpenSlide, x: int, y: int, patch_size: int, level: int = 0) -> Image.Image:
    return slide.read_region((int(x), int(y)), level, (patch_size, patch_size)).convert("RGB")


def is_tissue_patch(
    img: Image.Image,
    white_thr: int = 230,
    white_ratio_thr: float = 0.80,
    dark_thr: int = 25,
    dark_ratio_thr: float = 0.80
) -> bool:
    """
    Simple but effective tissue-region filtering.

    - High white ratio: usually blank slide background
    - High black ratio: usually borders, artifacts, or occlusion
    """
    arr = np.asarray(img, dtype=np.uint8)
    # White-region mask: all three channels are high
    white = (arr[:, :, 0] > white_thr) & (arr[:, :, 1] > white_thr) & (arr[:, :, 2] > white_thr)
    white_ratio = float(white.mean())

    # Black-region mask: all three channels are low
    dark = (arr[:, :, 0] < dark_thr) & (arr[:, :, 1] < dark_thr) & (arr[:, :, 2] < dark_thr)
    dark_ratio = float(dark.mean())

    if white_ratio > white_ratio_thr:
        return False
    if dark_ratio > dark_ratio_thr:
        return False
    return True


def crop_overview_to_tissue(
    thumb: Image.Image,
    pad_frac: float = 0.10,
    min_pad_px: int = 80,
    white_thr: int = 242,
) -> Image.Image:
    """
    Crop large blank margins on WSI thumbnail, keep all tissue islands (avoids ultra-wide canvas + tight bbox cropping left tissue).
    """
    arr = np.asarray(thumb)
    tissue = arr.mean(axis=2) < white_thr
    if not tissue.any():
        return thumb
    ys, xs = np.where(tissue)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = thumb.size
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    pad_x = max(min_pad_px, int(bw * pad_frac))
    pad_y = max(min_pad_px, int(bh * pad_frac))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w - 1, x1 + pad_x)
    y1 = min(h - 1, y1 + pad_y)
    return thumb.crop((x0, y0, x1 + 1, y1 + 1))


def _draw_boxes_on_thumb(
    thumb: Image.Image,
    coords,
    patch_size: int,
    scale_x: float,
    scale_y: float | None = None,
    l0_origin: tuple[int, int] = (0, 0),
    color=(255, 0, 0),
    width: int = 4,
) -> Image.Image:
    if scale_y is None:
        scale_y = scale_x
    ox, oy = l0_origin
    draw = ImageDraw.Draw(thumb)
    ref = max(thumb.size)
    line_w = max(2, min(20, int(round(width * ref / 2200))))
    for (x, y) in coords:
        x1 = int((x - ox) * scale_x)
        y1 = int((y - oy) * scale_y)
        x2 = int((x + patch_size - ox) * scale_x)
        y2 = int((y + patch_size - oy) * scale_y)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
    return thumb


def _draw_boxes_and_coord_labels(
    thumb: Image.Image,
    coords,
    patch_scores,
    patch_size: int,
    scale_x: float,
    scale_y: float,
    l0_origin: tuple[int, int],
    color=(255, 0, 0),
    width: int = 4,
    label_top_n: int | None = 4,
    font_scale: float = 1.0,
) -> Image.Image:
    """Fig.10 style: red boxes; label a(x,y)=score only on highest-scoring patches."""
    thumb = _draw_boxes_on_thumb(
        thumb, coords, patch_size, scale_x, scale_y, l0_origin, color=color, width=width
    )
    if not coords:
        return thumb
    draw = ImageDraw.Draw(thumb)
    base = max(11, min(15, int(12 * max(thumb.size) / 2800)))
    font = _load_font(max(8, int(round(base * font_scale))))
    ox, oy = l0_origin
    order = np.argsort(np.asarray(patch_scores, dtype=np.float64))[::-1]
    label_set = set(order[: max(1, int(label_top_n or 0))]) if label_top_n else set(range(len(coords)))
    for i, ((x, y), s) in enumerate(zip(coords, patch_scores)):
        y1 = int((float(y) - oy) * scale_y)
        x2 = int((float(x) + patch_size - ox) * scale_x)
        if i not in label_set:
            continue
        label = f"a({int(x)},{int(y)})={float(s):.2f}"
        tx = min(thumb.size[0] - 4, x2 + 4)
        ty = max(2, y1 - 2)
        tw = int(draw.textlength(label, font=font)) + 6
        th = max(16, int(round(16 * font_scale)))
        draw.rectangle([tx - 2, ty - 1, tx + tw, ty + th], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=(180, 20, 20), font=font)
    return thumb


def _scores_for_cluster(
    cluster_coords: list,
    all_coords: list,
    all_scores: list,
) -> list[float]:
    key_to_score = {
        (int(round(float(c[0]))), int(round(float(c[1])))): float(s)
        for c, s in zip(all_coords, all_scores)
    }
    return [
        key_to_score.get((int(round(float(c[0]))), int(round(float(c[1])))), 0.0)
        for c in cluster_coords
    ]


def tissue_bbox_level0(
    slide: openslide.OpenSlide,
    probe_long: int = 2048,
    white_thr: int = 242,
    pad_frac: float = 0.10,
) -> tuple[int, int, int, int]:
    """Fast low-res tissue bounding-box probe, mapped to level-0 coordinates."""
    w0, h0 = slide.level_dimensions[0]
    scale = probe_long / max(w0, h0)
    tw = max(1, int(w0 * scale))
    th = max(1, int(h0 * scale))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    arr = np.asarray(thumb)
    tissue = arr.mean(axis=2) < white_thr
    if not tissue.any():
        return 0, 0, w0, h0
    ys, xs = np.where(tissue)
    sx, sy = w0 / thumb.size[0], h0 / thumb.size[1]
    x0 = int(xs.min() * sx)
    y0 = int(ys.min() * sy)
    x1 = int((xs.max() + 1) * sx)
    y1 = int((ys.max() + 1) * sy)
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    return max(0, x0 - pad), max(0, y0 - pad), min(w0, x1 + pad), min(h0, y1 + pad)


def _union_l0_boxes(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def _coords_bbox_level0(coords, patch_size: int, pad_frac: float = 0.08) -> tuple[int, int, int, int]:
    xs = [int(x) for x, _ in coords] + [int(x) + patch_size for x, _ in coords]
    ys = [int(y) for _, y in coords] + [int(y) + patch_size for _, y in coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def _island_bboxes_from_thumb(
    thumb: Image.Image,
    white_thr: int = 242,
    min_cols: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Split disconnected tissue islands via column projection on thumbnail; return pixel bboxes (x0,y0,x1,y1)."""
    arr = np.asarray(thumb)
    col = (arr.mean(axis=2) < white_thr).sum(0)
    w = len(col)
    islands: list[tuple[int, int, int, int]] = []
    i = 0
    while i < w:
        while i < w and col[i] == 0:
            i += 1
        if i >= w:
            break
        j = i
        while j < w and col[j] > 0:
            j += 1
        if j - i >= min_cols:
            sub = arr[:, i:j]
            row = (sub.mean(axis=2) < white_thr).sum(1)
            ys = np.where(row > 0)[0]
            if len(ys):
                islands.append((i, int(ys.min()), j, int(ys.max()) + 1))
        i = j
    return islands


def tissue_island_bbox_level0(
    slide: openslide.OpenSlide,
    which: str = "left",
    probe_long: int = 2048,
) -> tuple[int, int, int, int]:
    """Return level-0 bbox of a single tissue island (left / right / largest)."""
    w0, h0 = slide.level_dimensions[0]
    scale = probe_long / max(w0, h0)
    tw = max(1, int(w0 * scale))
    th = max(1, int(h0 * scale))
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    islands = _island_bboxes_from_thumb(thumb)
    if not islands:
        return tissue_bbox_level0(slide, probe_long=probe_long)
    if which == "left":
        pick = min(islands, key=lambda b: b[0])
    elif which == "right":
        pick = max(islands, key=lambda b: b[2])
    else:
        pick = max(islands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    sx, sy = w0 / thumb.size[0], h0 / thumb.size[1]
    x0, y0, x1, y1 = pick
    pad = int(0.06 * max(x1 - x0, y1 - y0))
    return (
        max(0, int(x0 * sx) - pad),
        max(0, int(y0 * sy) - pad),
        min(w0, int(x1 * sx) + pad),
        min(h0, int(y1 * sy) + pad),
    )


def _split_patch_clusters(coords, gap_thr: float = 8000.0) -> list[list]:
    """Split top-k into spatial clusters by level-0 x gaps (left/right tissue islands, etc.)."""
    if len(coords) <= 1:
        return [list(coords)]
    pts = sorted(coords, key=lambda c: float(c[0]))
    clusters: list[list] = [[pts[0]]]
    for p in pts[1:]:
        if float(p[0]) - float(clusters[-1][-1][0]) > gap_thr:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return clusters


def _select_primary_cluster(
    coords: list,
    scores: list,
    gap_thr: float = 8000.0,
) -> list:
    """Fig.10 left overview uses main cluster only (most patches; tie-break by highest score) to avoid blurry distant outliers."""
    clusters = _split_patch_clusters(coords, gap_thr=gap_thr)
    if len(clusters) == 1:
        return clusters[0]

    def cluster_key(cl: list) -> tuple:
        cl_scores = _scores_for_cluster(cl, coords, scores)
        return (len(cl), max(cl_scores) if cl_scores else 0.0)

    return max(clusters, key=cluster_key)


def build_fig10_overview(
    slide: openslide.OpenSlide,
    coords: list,
    scores: list,
    patch_size: int,
    thumb_max_size: int = 7200,
    max_read_long_edge: int = 65536,
    pad_frac: float = 0.50,
    label_top_n: int = 4,
    font_scale: float = 2.0,
) -> Image.Image:
    """Fig.10 left: HD local overview of main tissue cluster + red boxes and coordinate labels for patches in that cluster."""
    primary = _select_primary_cluster(coords, scores)
    primary_scores = _scores_for_cluster(primary, coords, scores)
    return _overview_panel_from_cluster(
        slide,
        primary,
        patch_size,
        pad_frac=pad_frac,
        thumb_max_size=thumb_max_size,
        max_read_long_edge=max_read_long_edge,
        cluster_scores=primary_scores,
        draw_coord_labels=True,
        label_top_n=label_top_n,
        font_scale=font_scale,
    )


def _read_l0_roi_image(
    slide: openslide.OpenSlide,
    l0_box: tuple[int, int, int, int],
    thumb_max_size: int,
    max_read_long_edge: int,
) -> tuple[Image.Image, int, int, int, int]:
    w0, h0 = slide.level_dimensions[0]
    x0, y0, x1, y1 = l0_box
    rw, rh = max(1, x1 - x0), max(1, y1 - y0)

    best_lev = 0
    best_wh = (rw, rh)
    for lev in range(slide.level_count):
        try:
            down = float(slide.level_downsamples[lev])
        except Exception:
            down = max(w0 / max(slide.level_dimensions[lev][0], 1), 1.0)
        w_lev = max(1, int(np.ceil(rw / down)))
        h_lev = max(1, int(np.ceil(rh / down)))
        long_lev = max(w_lev, h_lev)
        if long_lev <= max_read_long_edge:
            best_lev = lev
            best_wh = (w_lev, h_lev)
            if long_lev >= thumb_max_size * 0.85:
                break

    w_lev, h_lev = best_wh
    thumb = slide.read_region((x0, y0), best_lev, (w_lev, h_lev)).convert("RGB")
    if max(thumb.size) > thumb_max_size:
        if thumb.size[0] >= thumb.size[1]:
            new_w = thumb_max_size
            new_h = max(1, int(thumb.size[1] * thumb_max_size / thumb.size[0]))
        else:
            new_h = thumb_max_size
            new_w = max(1, int(thumb.size[0] * thumb_max_size / thumb.size[1]))
        thumb = thumb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return thumb, x0, y0, rw, rh


def _overview_panel_from_cluster(
    slide: openslide.OpenSlide,
    cluster_coords: list,
    patch_size: int,
    pad_frac: float,
    thumb_max_size: int,
    max_read_long_edge: int,
    cluster_scores: list[float] | None = None,
    draw_coord_labels: bool = True,
    label_top_n: int = 4,
    color=(255, 0, 0),
    width: int = 4,
    font_scale: float = 1.0,
) -> Image.Image:
    w0, h0 = slide.level_dimensions[0]
    x0, y0, x1, y1 = _coords_bbox_level0(cluster_coords, patch_size, pad_frac=pad_frac)
    l0_box = (max(0, x0), max(0, y0), min(w0, x1), min(h0, y1))
    thumb, ox, oy, rw, rh = _read_l0_roi_image(
        slide, l0_box, thumb_max_size, max_read_long_edge
    )
    scale_x = thumb.size[0] / rw
    scale_y = thumb.size[1] / rh
    scores = cluster_scores if cluster_scores is not None else [1.0] * len(cluster_coords)
    if draw_coord_labels:
        return _draw_boxes_and_coord_labels(
            thumb,
            cluster_coords,
            scores,
            patch_size,
            scale_x,
            scale_y,
            l0_origin=(ox, oy),
            color=color,
            width=width,
            label_top_n=label_top_n,
            font_scale=font_scale,
        )
    return _draw_boxes_on_thumb(
        thumb,
        cluster_coords,
        patch_size,
        scale_x,
        scale_y,
        l0_origin=(ox, oy),
        color=color,
        width=width,
    )


def _stitch_overview_panels(
    panels: list[Image.Image],
    thumb_max_size: int,
    gap_px: int = 28,
) -> Image.Image:
    if len(panels) == 1:
        return panels[0]
    target_h = max(p.size[1] for p in panels)
    resized: list[Image.Image] = []
    for p in panels:
        if p.size[1] != target_h:
            nw = max(1, int(p.size[0] * target_h / p.size[1]))
            resized.append(p.resize((nw, target_h), Image.Resampling.LANCZOS))
        else:
            resized.append(p)
    pad = 12
    canvas_w = sum(p.size[0] for p in resized) + gap_px * (len(resized) - 1) + 2 * pad
    canvas_h = target_h + 2 * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    x = pad
    for p in resized:
        canvas.paste(p, (x, pad))
        x += p.size[0] + gap_px
    long_edge = max(canvas.size)
    if long_edge > thumb_max_size:
        scale = thumb_max_size / long_edge
        canvas = canvas.resize(
            (max(1, int(canvas.size[0] * scale)), max(1, int(canvas.size[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    return canvas


def _resolve_overview_l0_box(
    slide: openslide.OpenSlide,
    coords,
    patch_size: int,
    roi_mode: str,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    w0, h0 = slide.level_dimensions[0]
    if roi_mode == "patches":
        if not coords:
            raise ValueError("overview-roi=patches requires at least one patch coordinate")
        x0, y0, x1, y1 = _coords_bbox_level0(coords, patch_size, pad_frac=pad_frac)
    elif roi_mode in ("left", "right", "largest"):
        which = {"left": "left", "right": "right", "largest": "largest"}[roi_mode]
        x0, y0, x1, y1 = tissue_island_bbox_level0(slide, which=which)
        if coords:
            cx0, cy0, cx1, cy1 = _coords_bbox_level0(coords, patch_size, pad_frac=0.05)
            x0, y0 = max(x0, cx0), max(y0, cy0)
            x1, y1 = min(x1, cx1), min(y1, cy1)
    elif roi_mode == "tissue":
        x0, y0, x1, y1 = tissue_bbox_level0(slide)
        if coords:
            cx0, cy0, cx1, cy1 = _coords_bbox_level0(coords, patch_size, pad_frac=0.05)
            x0, y0 = min(x0, cx0), min(y0, cy0)
            x1, y1 = max(x1, cx1), max(y1, cy1)
    else:
        raise ValueError(f"Unknown overview-roi: {roi_mode}")
    return (
        max(0, x0),
        max(0, y0),
        min(w0, x1),
        min(h0, y1),
    )


def make_overview_thumbnail(
    slide: openslide.OpenSlide,
    coords,
    patch_size: int,
    thumb_max_size: int = 10000,
    max_read_long_edge: int = 65536,
    color=(255, 0, 0),
    width: int = 4,
    roi_mode: str = "patches",
    pad_frac: float = 0.45,
    stitch_clusters: bool = True,
    patch_scores: list[float] | None = None,
    draw_coord_labels: bool = True,
) -> Image.Image:
    """
    Read high-resolution tissue ROI.
    In patches mode, if top-k spans multiple spatial clusters, stitch clusters horizontally by default (keeps high-score patches on the right).
    """
    if roi_mode == "patches" and stitch_clusters:
        clusters = _split_patch_clusters(coords)
        if len(clusters) > 1:
            per_panel_max = max(4000, int(thumb_max_size * 0.72))
            panels = []
            for cl in clusters:
                cl_scores = (
                    _scores_for_cluster(cl, coords, patch_scores)
                    if patch_scores is not None
                    else None
                )
                panels.append(
                    _overview_panel_from_cluster(
                        slide,
                        cl,
                        patch_size,
                        pad_frac,
                        per_panel_max,
                        max_read_long_edge,
                        cluster_scores=cl_scores,
                        draw_coord_labels=draw_coord_labels,
                        color=color,
                        width=width,
                    )
                )
            return _stitch_overview_panels(panels, thumb_max_size)

    w0, h0 = slide.level_dimensions[0]
    x0, y0, x1, y1 = _resolve_overview_l0_box(
        slide, coords, patch_size, roi_mode=roi_mode, pad_frac=pad_frac
    )
    thumb, ox, oy, rw, rh = _read_l0_roi_image(
        slide, (x0, y0, x1, y1), thumb_max_size, max_read_long_edge
    )
    scale_x = thumb.size[0] / rw
    scale_y = thumb.size[1] / rh
    if draw_coord_labels and patch_scores is not None:
        return _draw_boxes_and_coord_labels(
            thumb,
            coords,
            patch_scores,
            patch_size,
            scale_x,
            scale_y,
            l0_origin=(ox, oy),
            color=color,
            width=width,
        )
    return _draw_boxes_on_thumb(
        thumb,
        coords,
        patch_size,
        scale_x,
        scale_y,
        l0_origin=(ox, oy),
        color=color,
        width=width,
    )


def save_overview_only(
    thumb: Image.Image,
    out_png: str,
    title: str = "",
    sharpen: bool = True,
) -> None:
    thumb = thumb.convert("RGB")
    if sharpen:
        from PIL import ImageFilter

        thumb = thumb.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=2))

    title_h = 0
    if title:
        title_h = 44
        font = _load_font(20)
        canvas = Image.new("RGB", (thumb.size[0] + 24, thumb.size[1] + title_h + 16), (255, 255, 255))
        canvas.paste(thumb, (12, title_h))
        ImageDraw.Draw(canvas).text((12, 8), title, fill=(17, 24, 39), font=font)
        out_img = canvas
    else:
        out_img = thumb

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    out_img.save(out_png, format="PNG", compress_level=1)
    print(f"[ok] overview only -> {out_png} ({out_img.size[0]}x{out_img.size[1]})")


def _pick_thumbnail_level(
    slide: openslide.OpenSlide, thumb_long_edge: int, max_read_long_edge: int
):
    """
    Return pyramid level for full-layer read_region; None if too large/unavailable (fallback to get_thumbnail).
    """
    dims = slide.level_dimensions
    if not dims:
        return 0
    safe = [lev for lev in range(len(dims)) if max(dims[lev]) <= max_read_long_edge]
    if not safe:
        return None
    good = [lev for lev in safe if max(dims[lev]) >= thumb_long_edge]
    if good:
        return int(min(good))
    return int(min(safe))


def make_thumbnail_with_boxes(
    slide,
    coords,
    patch_size,
    thumb_max_size=4096,
    color=(255, 0, 0),
    width=4,
    max_read_long_edge: int = 16384,
):
    """
    Generate left WSI overview with box annotations.

    - Default ``thumb_max_size`` long edge raised to 4096, much sharper than legacy 1800.
    - Pick pyramid level for full ``read_region``, then LANCZOS downscale if needed;
      Fallback to ``get_thumbnail`` when level-0 is too large, still using a large target long edge.
    - Coordinates and ``patch_size`` are in level-0 pixel space (consistent with read_patch(level=0)).
    """
    w0, h0 = slide.level_dimensions[0]
    long0 = max(w0, h0)

    def draw_on_thumb(thumb: Image.Image, scale_level0_to_thumb: float) -> Image.Image:
        return _draw_boxes_on_thumb(
            thumb,
            coords,
            patch_size,
            scale_level0_to_thumb,
            color=color,
            width=width,
        )

    # Oversized level-0: avoid reading full slide at once; use OpenSlide downscale chain
    if long0 > max_read_long_edge:
        scale = min(thumb_max_size / w0, thumb_max_size / h0)
        thumb_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGB")
        return draw_on_thumb(thumb, scale)

    lev = _pick_thumbnail_level(slide, thumb_max_size, max_read_long_edge)
    if lev is None:
        scale = min(thumb_max_size / w0, thumb_max_size / h0)
        thumb_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))
        thumb = slide.get_thumbnail(thumb_size).convert("RGB")
        return draw_on_thumb(thumb, scale)

    w_lev, h_lev = slide.level_dimensions[lev]
    try:
        down = float(slide.level_downsamples[lev])
    except Exception:
        down = max(w0 / max(w_lev, 1), h0 / max(h_lev, 1))

    thumb = slide.read_region((0, 0), lev, (w_lev, h_lev)).convert("RGB")
    scale_lev_to_thumb = 1.0
    long_lev = max(w_lev, h_lev)
    if long_lev > thumb_max_size:
        if w_lev >= h_lev:
            new_w, new_h = thumb_max_size, max(1, int(h_lev * thumb_max_size / w_lev))
        else:
            new_h, new_w = thumb_max_size, max(1, int(w_lev * thumb_max_size / h_lev))
        thumb = thumb.resize((new_w, new_h), Image.Resampling.LANCZOS)
        scale_lev_to_thumb = new_w / w_lev

    # level-0 -> thumb: level-0 -> lev, then lev -> thumb (if resize)
    scale_level0_to_thumb = (1.0 / down) * scale_lev_to_thumb
    return draw_on_thumb(thumb, scale_level0_to_thumb)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _patch_with_red_border(patch: Image.Image, cell: int, border: int = 3) -> Image.Image:
    inner = patch.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (cell + 2 * border, cell + 2 * border), (220, 20, 20))
    out.paste(inner, (border, border))
    return out


def plot_figure_fig10(
    overview_img: Image.Image,
    patch_imgs: list,
    patch_coords: list,
    patch_scores: list,
    patch_node_ids: list[int] | None,
    header_line1: str,
    header_line2: str,
    out_png: str,
    grid: int = 3,
    prob_mutant: float | None = None,
    paper_export: bool = True,
    show_header: bool = False,
    show_legend: bool = False,
    font_scale: float | None = None,
) -> None:
    """Paper Fig.10 style; paper_export uses concise #k (score) legend on the right; captions handled by LaTeX by default."""
    if font_scale is None:
        font_scale = 2.0 if paper_export else 1.0
    from PIL import ImageFilter

    overview = overview_img.convert("RGB")
    if paper_export:
        overview = overview.filter(ImageFilter.UnsharpMask(radius=0.8, percent=85, threshold=3))

    cell = 400 if paper_export else 380
    border = 4
    cell_outer = cell + 2 * border
    gap = 14
    label_h = int(round(28 * font_scale))
    row_label_w = 0
    grid_title_h = int(round(30 * font_scale))
    header_h = 8
    if show_header:
        header_h = 62 if (header_line1 or header_line2) else 8
        if prob_mutant is not None:
            header_h += 22

    grid_w = row_label_w + grid * cell_outer + (grid - 1) * gap
    grid_h = grid_title_h + grid * (cell_outer + label_h) + (grid - 1) * gap

    target_oh = grid_h
    max_ow = int(target_oh * 1.22)
    scale = min(target_oh / max(overview.size[1], 1), max_ow / max(overview.size[0], 1))
    target_ow = max(1, int(overview.size[0] * scale))
    target_oh = max(1, int(overview.size[1] * scale))
    overview = overview.resize((target_ow, target_oh), Image.Resampling.LANCZOS)

    pad = 22
    gap_lr = 40
    legend_h = int(round(70 * font_scale)) if (paper_export and show_legend) else 0
    canvas_w = pad + target_ow + gap_lr + grid_w + pad
    canvas_h = pad + header_h + max(target_oh, grid_h) + pad + legend_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_hdr = _load_font(int(round((26 if paper_export else 24) * font_scale)))
    font_sub = _load_font(int(round(18 * font_scale)))
    font_lbl = _load_font(int(round((17 if paper_export else 16) * font_scale)))
    font_leg = _load_font(int(round(14 * font_scale)))
    font_grid_title = _load_font(int(round(19 * font_scale)))

    if show_header:
        y_hdr = pad
        if header_line1:
            draw.text((pad, y_hdr), header_line1, fill=(20, 20, 20), font=font_hdr)
            y_hdr += 30
        if header_line2:
            draw.text((pad, y_hdr), header_line2, fill=(60, 60, 60), font=font_sub)
            y_hdr += 26
        if prob_mutant is not None:
            draw.text(
                (pad, y_hdr),
                f"P(mutant) = {prob_mutant:.3f}  |  attribution target: mutant class (y=1)",
                fill=(80, 80, 80),
                font=font_sub,
            )

    oy = pad + header_h + max(0, (grid_h - target_oh) // 2)
    canvas.paste(overview, (pad, oy))

    gx0 = pad + target_ow + gap_lr
    gy0 = pad + header_h
    draw.text(
        (gx0 + row_label_w, gy0),
        "Top-9 patches (ranked by gradient attribution)",
        fill=(30, 30, 30),
        font=font_grid_title,
    )
    gy0 += grid_title_h

    n = min(len(patch_imgs), grid * grid)
    for i in range(n):
        row, col = i // grid, i % grid
        px = gx0 + row_label_w + col * (cell_outer + gap)
        py = gy0 + row * (cell_outer + label_h + gap)
        bordered = _patch_with_red_border(patch_imgs[i], cell, border=border)
        canvas.paste(bordered, (px, py))
        s = patch_scores[i]
        if paper_export:
            cap = f"#{i + 1}  (s={float(s):.2f})"
        else:
            x, y = patch_coords[i]
            cap = f"a({int(x)},{int(y)})={float(s):.2f}"
        tw = draw.textlength(cap, font=font_lbl)
        draw.text((px + max(0, (cell_outer - tw) / 2), py + cell_outer + 4), cap, fill=(35, 35, 35), font=font_lbl)

    if paper_export and show_legend:
        ly = canvas_h - pad - legend_h + 8
        lx = pad
        draw.rectangle([lx, ly, lx + target_ow - 8, ly + legend_h - 10], outline=(210, 210, 210), width=1)
        draw.rectangle([lx + 12, ly + 16, lx + 28, ly + 32], outline=(200, 20, 20), width=2)
        draw.text(
            (lx + 36, ly + 12),
            "Red box: top-K patch on WSI (coordinates a(x,y) on overview)",
            fill=(50, 50, 50),
            font=font_leg,
        )
        draw.text(
            (lx + 36, ly + 34),
            "Right: 256x256 crops; #k = rank, s = attribution score (slide-robust or top-k spread)",
            fill=(50, 50, 50),
            font=font_leg,
        )

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png, format="PNG", compress_level=1)
    print(f"[ok] Fig.10 layout -> {out_png} ({canvas_w}x{canvas_h})")


def plot_figure_stitch(
    thumb_img: Image.Image,
    patch_imgs,
    patch_coords,
    patch_scores,
    title: str,
    out_png: str,
    grid: int = 3,
    paper_style: bool = False,
) -> None:
    """PIL horizontal stitch: ultra-wide WSI not cropped by matplotlib axes."""
    thumb = thumb_img.convert("RGB")
    tw, th = thumb.size
    cell = max(220, min(320, th // grid))
    gap = 28
    title_h = 52 if title else 12
    grid_w = grid * cell + (grid - 1) * 8
    grid_h = grid * cell + (grid - 1) * 8 + 22
    canvas_w = tw + gap + grid_w + 24
    canvas_h = max(th, grid_h) + title_h + 16
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    y0 = title_h + max(0, (canvas_h - title_h - th) // 2)
    canvas.paste(thumb, (12, y0))

    gx0 = 12 + tw + gap
    gy0 = title_h + max(0, (canvas_h - title_h - grid_h) // 2)
    font_t = _load_font(18 if not paper_style else 16)
    font_s = _load_font(14 if not paper_style else 12)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((12, 10), title, fill=(17, 24, 39), font=font_t)

    for i in range(grid * grid):
        row, col = i // grid, i % grid
        x = gx0 + col * (cell + 8)
        y = gy0 + row * (cell + 8 + 22)
        if i < len(patch_imgs):
            patch = patch_imgs[i].convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(patch, (x, y))
            s = patch_scores[i]
            if paper_style:
                label = f"#{i + 1}  ({s:.2f})"
            else:
                px, py = patch_coords[i]
                label = f"a({int(px)},{int(py)})={s:.2f}"
            draw.text((x, y - 20), label, fill=(55, 65, 81), font=font_s)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png, format="PNG", optimize=True)
    print(f"[ok] stitch layout -> {out_png} ({canvas_w}x{canvas_h})")


def plot_figure(
    thumb_img,
    patch_imgs,
    patch_coords,
    patch_scores,
    title,
    out_png,
    grid=3,
    save_dpi: int = 300,
    paper_style: bool = False,
):
    if hasattr(thumb_img, "size"):
        tw, th = thumb_img.size
    else:
        tw, th = int(thumb_img.shape[1]), int(thumb_img.shape[0])
    thumb_aspect = max(tw / max(th, 1), 0.25)
    if thumb_aspect >= 1.75:
        return plot_figure_stitch(
            thumb_img,
            patch_imgs,
            patch_coords,
            patch_scores,
            title,
            out_png,
            grid=grid,
            paper_style=paper_style,
        )
    patch_panel_in = max(3.8, grid * 1.35)
    overview_h_in = patch_panel_in
    overview_w_in = overview_h_in * thumb_aspect
    gap_in = 0.35
    margin_l_in, margin_b_in, margin_t_in = 0.35, 0.40, 0.85
    fig_w = margin_l_in + overview_w_in + gap_in + patch_panel_in + 0.35
    fig_h = margin_b_in + overview_h_in + 0.35

    if paper_style:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.titlesize": 9,
                "figure.facecolor": "white",
            }
        )

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=save_dpi, facecolor="white")

    # Inch-level layout: WSI axis width = height * aspect ratio to avoid gridspec squeezing wide slides
    ax0 = fig.add_axes(
        [
            margin_l_in / fig_w,
            margin_b_in / fig_h,
            overview_w_in / fig_w,
            overview_h_in / fig_h,
        ]
    )
    arr = np.asarray(thumb_img)
    ax0.imshow(arr, aspect="equal", interpolation="bilinear", clip_on=False)
    ax0.set_xlim(-0.5, tw - 0.5)
    ax0.set_ylim(th - 0.5, -0.5)
    ax0.set_aspect("equal", adjustable="box")
    if title:
        ax0.set_title(title, fontsize=11 if paper_style else 12, pad=4, y=1.02)
    ax0.axis("off")

    patch_x0 = (margin_l_in + overview_w_in + gap_in) / fig_w
    patch_w = patch_panel_in / fig_w
    patch_h = patch_panel_in / fig_h
    patch_y0 = margin_b_in / fig_h + (overview_h_in - patch_panel_in) / (2 * fig_h)
    cell_w = patch_w / grid
    cell_h = patch_h / grid
    for i in range(grid * grid):
        row, col = i // grid, i % grid
        ax = fig.add_axes(
            [
                patch_x0 + col * cell_w,
                patch_y0 + (grid - 1 - row) * cell_h,
                cell_w * 0.94,
                cell_h * 0.88,
            ]
        )
        ax.axis("off")
        if i < len(patch_imgs):
            ax.imshow(patch_imgs[i])
            s = patch_scores[i]
            if paper_style:
                ax.set_title(f"#{i + 1}  ({s:.2f})", fontsize=8, pad=2)
            else:
                x, y = patch_coords[i]
                ax.set_title(f"a({int(x)},{int(y)})={s:.2f}", fontsize=9)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=save_dpi, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to graph .pt file; must contain data.pos")
    ap.add_argument("--wsi", required=True, help="Original WSI path, e.g. .svs or .tif")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--out", default="wsi_top_patches.png")
    ap.add_argument("--cls", type=int, default=1)
    ap.add_argument("--method", choices=["grad_x_input", "grad_norm"], default="grad_x_input")
    ap.add_argument("--topk", type=int, default=9, help="Number of top patches to display")
    ap.add_argument("--patch_size", type=int, default=256, help="Patch size at level 0")
    ap.add_argument(
        "--thumb_max",
        type=int,
        default=4096,
        help="Target long-edge pixels for left WSI thumbnail (larger = sharper, more memory/disk).",
    )
    ap.add_argument(
        "--save-dpi",
        type=int,
        default=300,
        help="Matplotlib DPI for PNG export; combined with figsize sets total pixels.",
    )
    ap.add_argument(
        "--max-read-long-edge",
        type=int,
        default=16384,
        help="Max long edge for single-layer read_region; fallback to get_thumbnail to avoid OOM.",
    )
    ap.add_argument("--device", default="cuda:0")

    # Background filtering parameters
    ap.add_argument("--white_thr", type=int, default=230)
    ap.add_argument("--white_ratio_thr", type=float, default=0.80)
    ap.add_argument("--dark_thr", type=int, default=25)
    ap.add_argument("--dark_ratio_thr", type=float, default=0.80)

    # Try more high-score candidates when filtering is strict
    ap.add_argument("--max_candidates", type=int, default=2000, help="Max high-score candidate nodes to try")
    ap.add_argument(
        "--paper-style",
        action="store_true",
        help="Paper mode: title omits scale_w; patch labels show rank not coordinates.",
    )
    ap.add_argument(
        "--panel-label",
        type=str,
        default="",
        help="Optional panel label, e.g. '(a)', shown at top-left of WSI subplot.",
    )
    ap.add_argument(
        "--cohort-gene",
        type=str,
        default="",
        help="Cohort/gene in paper title, e.g. 'TCGA-THYM (GTF2I)'.",
    )
    ap.add_argument(
        "--label-status",
        type=str,
        default="",
        help="Phenotype in paper title, e.g. 'mutation-positive'.",
    )
    ap.add_argument(
        "--no-crop-overview",
        action="store_true",
        help="Keep full-slide thumbnail (default crops to tissue bbox to avoid cutting left tissue).",
    )
    ap.add_argument(
        "--overview-only",
        action="store_true",
        help="Export left WSI overview only (no right patch grid), higher resolution by default.",
    )
    ap.add_argument(
        "--no-overview-title",
        action="store_true",
        help="Omit top title on overview-only export.",
    )
    ap.add_argument(
        "--overview-roi",
        choices=("patches", "left", "right", "largest", "tissue"),
        default="patches",
        help="Overview crop: patches=around top-k; left/right/largest=single island; tissue=all tissue.",
    )
    ap.add_argument(
        "--overview-pad",
        type=float,
        default=0.45,
        help="Padding ratio around top-k bbox when overview-roi=patches.",
    )
    ap.add_argument(
        "--no-stitch-clusters",
        action="store_true",
        help="When top-k spans clusters, keep largest cluster only (drops distant high-score patches).",
    )
    ap.add_argument(
        "--layout",
        choices=("fig10", "classic"),
        default="fig10",
        help="fig10=paper Fig.10 left overview + right patch grid; classic=legacy matplotlib layout.",
    )
    ap.add_argument(
        "--slide-id",
        type=str,
        default="",
        help="Second-line sample ID at overview top-left (default: WSI filename stem).",
    )
    ap.add_argument(
        "--mutation-label",
        type=str,
        default="",
        help="First line at overview top-left, e.g. 'BRAF-Mutation' (default from --cohort-gene).",
    )
    ap.add_argument(
        "--fig10-header",
        action="store_true",
        help="Draw sample/probability text in Fig.10 layout (off by default; use LaTeX caption).",
    )
    ap.add_argument(
        "--show-legend",
        action="store_true",
        help="Draw explanation box in Fig.10 layout (off for paper export; use LaTeX caption).",
    )
    ap.add_argument(
        "--score-display",
        choices=("paper", "robust", "percentile", "topk_spread"),
        default="paper",
        help="Paper score display: paper=slide-wide robust; spread within top-k if scores cluster.",
    )
    ap.add_argument(
        "--overview-label-top",
        type=int,
        default=4,
        help="Label a(x,y)=s only on top N patches in left overview (others red box only).",
    )
    ap.add_argument(
        "--font-scale",
        type=float,
        default=2.0,
        help="Text scale for paper export (default 2.0 for LaTeX full-width readability).",
    )
    ap.add_argument(
        "--overview-font-scale",
        type=float,
        default=None,
        help="Font size for coordinate labels on left WSI overview (default font-scale×2.5).",
    )

    args = ap.parse_args()
    if args.overview_font_scale is None:
        args.overview_font_scale = args.font_scale * 2.5
    if args.overview_only:
        if args.thumb_max == 4096:
            args.thumb_max = 10000
        if args.max_read_long_edge == 16384:
            args.max_read_long_edge = 65536
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    data = load_graph(args.pt, device)
    encoder, classifier = load_models(args.encoder, args.classifier, device)

    scores_raw, prob, scale_w = grad_importance(encoder, classifier, data, args.cls, args.method)
    pos = data.pos.detach().cpu().numpy()
    coords_all = pos[:, :2]

    # Sort by raw gradient strength descending (no per-slide min-max to avoid tied top scores)
    order = np.argsort(scores_raw)[::-1]

    slide = openslide.OpenSlide(args.wsi)

    # Try candidates one by one; collect top-k patches after background filtering
    selected_coords = []
    selected_scores = []
    selected_patches = []
    selected_node_ids: list[int] = []

    tried = 0
    for idx in order[: min(len(order), args.max_candidates)]:
        tried += 1
        x, y = coords_all[idx]
        patch = read_patch(slide, x, y, args.patch_size, level=0)

        if not is_tissue_patch(
            patch,
            white_thr=args.white_thr,
            white_ratio_thr=args.white_ratio_thr,
            dark_thr=args.dark_thr,
            dark_ratio_thr=args.dark_ratio_thr
        ):
            continue

        selected_coords.append((x, y))
        selected_scores.append(float(scores_raw[idx]))
        selected_patches.append(patch)
        selected_node_ids.append(int(idx))

        if len(selected_coords) >= args.topk:
            break

    display_scores = format_attribution_scores(
        scores_raw, selected_scores, mode=args.score_display
    )

    if len(selected_coords) == 0:
        raise RuntimeError(
            "No valid patches selected: filters may be too strict, or pos/patch_size/coordinate system mismatch."
            "Try relaxing --white_ratio_thr or increasing --max_candidates."
        )

    if args.overview_only:
        thumb = make_overview_thumbnail(
            slide=slide,
            coords=selected_coords,
            patch_size=args.patch_size,
            thumb_max_size=args.thumb_max,
            max_read_long_edge=args.max_read_long_edge,
            roi_mode=args.overview_roi,
            pad_frac=args.overview_pad,
            stitch_clusters=not args.no_stitch_clusters,
        )
    else:
        use_fig10 = args.layout == "fig10"
        if use_fig10:
            # Left: main tissue cluster only; right 3×3 still shows all top-k (including other clusters)
            primary_coords = _select_primary_cluster(selected_coords, display_scores)
            primary_display = format_attribution_scores(
                scores_raw,
                _scores_for_cluster(primary_coords, selected_coords, display_scores),
                mode=args.score_display,
            )
            thumb = build_fig10_overview(
                slide=slide,
                coords=primary_coords,
                scores=primary_display,
                patch_size=args.patch_size,
                thumb_max_size=max(args.thumb_max, 7200),
                max_read_long_edge=max(args.max_read_long_edge, 65536),
                pad_frac=args.overview_pad,
                label_top_n=max(1, args.overview_label_top),
                font_scale=args.overview_font_scale,
            )
        else:
            thumb = make_overview_thumbnail(
                slide=slide,
                coords=selected_coords,
                patch_size=args.patch_size,
                thumb_max_size=args.thumb_max,
                max_read_long_edge=args.max_read_long_edge,
                roi_mode="patches",
                pad_frac=args.overview_pad,
                stitch_clusters=not args.no_stitch_clusters,
                patch_scores=display_scores,
                draw_coord_labels=False,
            )

    if args.paper_style:
        parts = []
        if args.panel_label:
            parts.append(args.panel_label.strip())
        if args.cohort_gene:
            parts.append(args.cohort_gene.strip())
        if args.label_status:
            parts.append(args.label_status.strip())
        parts.append(f"$P(\\mathrm{{mutant}})={prob:.3f}$")
        title = ", ".join(parts) if parts else f"$P(\\mathrm{{mutant}})={prob:.3f}$"
    else:
        title = f"Top-{len(selected_coords)} patches (tissue-filtered) | target_cls={args.cls} prob={prob:.3f}"
        if scale_w is not None and len(scale_w) >= 2:
            title += f" | scale_w(patch,region)=({scale_w[0]:.2f},{scale_w[1]:.2f})"

    if args.overview_only:
        out_title = "" if args.no_overview_title else title
        save_overview_only(thumb, args.out, title=out_title)
    else:
        grid = int(np.ceil(np.sqrt(args.topk)))
        if args.layout == "fig10":
            wsi_stem = Path(args.wsi).stem
            slide_id = args.slide_id.strip() or wsi_stem
            mut_label = args.mutation_label.strip()
            if not mut_label and args.cohort_gene:
                mut_label = (
                    args.cohort_gene.replace("TCGA-", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "-")
                    + "-Mutation"
                )
            if not mut_label:
                mut_label = f"target_cls={args.cls}"
            plot_figure_fig10(
                overview_img=thumb,
                patch_imgs=selected_patches,
                patch_coords=selected_coords,
                patch_scores=display_scores,
                patch_node_ids=selected_node_ids,
                header_line1=mut_label,
                header_line2=slide_id,
                out_png=args.out,
                grid=grid,
                prob_mutant=prob,
                paper_export=True,
                show_header=args.fig10_header,
                show_legend=args.show_legend,
                font_scale=args.font_scale,
            )
        else:
            plot_figure(
                thumb_img=thumb,
                patch_imgs=selected_patches,
                patch_coords=selected_coords,
                patch_scores=selected_scores,
                title=title,
                out_png=args.out,
                grid=grid,
                save_dpi=args.save_dpi,
                paper_style=args.paper_style,
            )

    print(f"[Done] Saved to: {args.out}")
    print(f"[Info] Tried {tried} candidates, kept {len(selected_coords)}")


if __name__ == "__main__":
    main()
