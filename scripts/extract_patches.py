"""
Tile WSIs into tissue patches and save them as JPEG files.

Features:
- Tissue detection on a low-resolution thumbnail
- Automatic pyramid level and patch-size selection
- Recursive processing of .svs / .tif / .tiff files

Output layout:
    data/TCGA-{DATASET}/patches/{slide_name}/{slide_name}_{x}_{y}.jpg
"""

from __future__ import annotations

import argparse
import datetime
import gc
import os
from pathlib import Path

import cv2
import numpy as np
import openslide
from openslide import OpenSlideError
from PIL import Image
from tqdm import tqdm


def get_tissue_mask(slide, thumb_max_size=2000):
    """Build a tissue mask from a thumbnail; returns mask and level-0 scale factors."""
    base_w, base_h = slide.level_dimensions[0]

    scale = thumb_max_size / max(base_w, base_h)
    thumb_w = max(1, int(base_w * scale))
    thumb_h = max(1, int(base_h * scale))

    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    thumb_np = np.array(thumb)
    thumb_hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)

    _, saturation_mask = cv2.threshold(
        thumb_hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    mask = cv2.morphologyEx(saturation_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    scale_x = thumb_w / base_w
    scale_y = thumb_h / base_h
    return mask, scale_x, scale_y


def judge_save_and_write(region, new_path, svs_name, x_point, y_point):
    """Filter out mostly white or mostly dark patches before saving."""
    try:
        image_rgb = region.convert("RGB")
        image_gray = image_rgb.convert("L")
        image_np = np.array(image_gray)

        h, w = image_np.shape
        white_pixel_count = np.sum(image_np > 230)
        dark_pixel_count = np.sum(image_np < 30)
        denom = max(1, h * w)

        if (white_pixel_count / denom) < 0.6 and (dark_pixel_count / denom) < 0.3:
            img_name = f"{svs_name}_{x_point}_{y_point}.jpg"
            image_rgb.save(os.path.join(new_path, img_name), quality=95)
            return True
        return False
    except Exception as exc:
        print(f"[warn] patch quality filter failed: {exc}")
        return False


def deal_patches_streaming(
    slide,
    patch_path,
    svs_name,
    target_min=6000,
    target_max=25000,
    base_patch_size=224,
):
    """Select pyramid level and patch size automatically, then stream patches to disk."""
    mask, scale_x, scale_y = get_tissue_mask(slide)
    mask_h, mask_w = mask.shape

    def estimate_patches(level):
        w, h = slide.level_dimensions[level]
        return (w // base_patch_size) * (h // base_patch_size)

    levels = list(range(len(slide.level_dimensions)))
    estimates = [(level, estimate_patches(level)) for level in levels]

    mid_target = (target_min + target_max) / 2
    estimates_sorted = sorted(estimates, key=lambda item: abs(item[1] - mid_target))
    cut_level = estimates_sorted[0][0]

    if cut_level == len(slide.level_dimensions) - 1:
        cut_level = max(0, cut_level - 1)

    cut_w, cut_h = slide.level_dimensions[cut_level]
    cut_ds = slide.level_downsamples[cut_level]

    patch_est = (cut_w // base_patch_size) * (cut_h // base_patch_size)
    if patch_est < target_min:
        patch_size = int(base_patch_size * 0.75)
    elif patch_est > target_max:
        patch_size = int(base_patch_size * 1.4)
    else:
        patch_size = base_patch_size

    xs = list(range(0, cut_w, patch_size))
    ys = list(range(0, cut_h, patch_size))

    new_path = os.path.join(patch_path, svs_name)
    os.makedirs(new_path, exist_ok=True)

    saved = 0
    pbar = tqdm(total=len(xs) * len(ys), desc=f"Tiles [{svs_name}]", ncols=90)

    for y in ys:
        for x in xs:
            base_x = x * cut_ds
            base_y = y * cut_ds
            mask_x = int(base_x * scale_x)
            mask_y = int(base_y * scale_y)

            if (
                0 <= mask_x < mask_w
                and 0 <= mask_y < mask_h
                and mask[mask_y, mask_x] > 0
            ):
                region = slide.read_region((x, y), cut_level, (patch_size, patch_size))
                if judge_save_and_write(region, new_path, svs_name, x, y):
                    saved += 1

            pbar.update(1)

    pbar.close()
    return saved


def extract_patches(dataset_dir: Path, wsi_dir: Path) -> None:
    print("-" * 42)
    print(f"Dataset dir: {dataset_dir}")
    print(f"WSI dir:     {wsi_dir}")
    print("-" * 42)

    patch_path = dataset_dir / "patches"
    patch_path.mkdir(parents=True, exist_ok=True)

    svs_list = [
        Path(root) / fname
        for root, _, files in os.walk(wsi_dir)
        for fname in files
        if fname.lower().endswith((".svs", ".tif", ".tiff"))
    ]

    for svs_path in tqdm(svs_list, desc="WSI files", ncols=90):
        svs_name = svs_path.name.split(".")[0]
        start_time = datetime.datetime.now()

        try:
            slide = openslide.open_slide(str(svs_path))
        except Exception as exc:
            print(f"[warn] cannot open slide {svs_name}: {exc}")
            continue

        try:
            patches_num = deal_patches_streaming(slide, str(patch_path), svs_name)
            cost = (datetime.datetime.now() - start_time).seconds
            print(f"  - {svs_name}: {patches_num} patches, {cost}s")
        except Exception as exc:
            print(f"[error] failed on {svs_name}: {exc}")
        finally:
            slide.close()
            gc.collect()

    print("-" * 42)
    print("Patch extraction complete")
    print("-" * 42)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile WSIs into patch JPEG files.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root, e.g. data/TCGA-THYM")
    parser.add_argument("--wsi-dir", default=None, help="WSI directory (default: <dataset-dir>/wsi)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    wsi_dir = Path(args.wsi_dir or dataset_dir / "wsi")
    extract_patches(dataset_dir, wsi_dir)


if __name__ == "__main__":
    main()
