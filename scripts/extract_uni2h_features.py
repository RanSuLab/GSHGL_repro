"""
Extract UNI-2h patch embeddings and aggregate them per WSI.

Expected input layout:
    data/TCGA-{DATASET}/patches/
        TCGA-XX-XXXX-01A-01-TSA_4928_16352.jpg
        ...

Output:
    data/TCGA-{DATASET}/data_features.pkl
        {
            "names_list":    [slide_id, ...],
            "coords_list":   [ndarray(N, 2), ...],
            "features_list": [ndarray(N, 1536), ...],
        }

UNI-2h weights are gated on Hugging Face. Request access at:
https://huggingface.co/MahmoodLab/UNI2-h
Then authenticate before the first run:
    huggingface-cli login
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class PatchDataset(Dataset):
    def __init__(self, image_paths, coords, names, transform):
        self.image_paths = image_paths
        self.coords = coords
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        coord = self.coords[idx]
        name = self.names[idx]
        return img, np.array(coord, dtype=np.int32), name


def load_model_and_transforms(device=None, assets_dir: str | Path = "assets/ckpts"):
    """Load UNI-2h (ViT-Giant, 1536-dim) and its ImageNet normalization transform."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    enc_name = "uni2-h"
    checkpoint_file = "pytorch_model.bin"
    ckpt_dir = Path(assets_dir) / enc_name
    ckpt_path = ckpt_dir / checkpoint_file

    if not ckpt_path.is_file():
        logger.info(f"Checkpoint not found at {ckpt_path}; downloading from Hugging Face")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id="MahmoodLab/UNI2-h",
                filename=checkpoint_file,
                local_dir=str(ckpt_dir),
                force_download=True,
            )
            logger.info("Download complete")
        except Exception as exc:
            logger.exception("Hugging Face download failed; confirm you are logged in and approved")
            logger.info("Run: huggingface-cli login")
            raise exc

    uni_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(**uni_kwargs)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


@torch.no_grad()
def extract_features_from_patches(
    model,
    patches_dir: Path,
    transform,
    device=None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Walk ``patches_dir`` and encode all ``.jpg`` patches.

    Filename convention:
        {slide_prefix}_{x}_{y}.jpg
    e.g. TCGA-4V-A9QS-01A-01-TSA_4928_16352.jpg
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths, coords, names = [], [], []
    for root, _, files in os.walk(patches_dir):
        for fname in files:
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("_")
            if len(parts) < 3 or not parts[-1].endswith(".jpg"):
                continue
            try:
                x = int(parts[-2])
                y = int(parts[-1].replace(".jpg", ""))
                name_parts = parts[0].split("-")
                if len(name_parts) < 3:
                    continue
                name = f"{name_parts[0]}-{name_parts[1]}-{name_parts[2]}"
                image_paths.append(os.path.join(root, fname))
                coords.append([x, y])
                names.append(name)
            except (ValueError, IndexError):
                continue

    if not image_paths:
        raise FileNotFoundError(f"No .jpg patches found under {patches_dir}")

    logger.info(f"Patches to encode: {len(image_paths)}")
    dataset = PatchDataset(image_paths, coords, names, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features_list, coords_list, names_list = [], [], []
    for imgs, batch_coords, batch_names in tqdm(loader, desc="UNI-2h"):
        imgs = imgs.to(device)
        with torch.inference_mode():
            batch_features = model(imgs).detach().cpu().numpy()
        features_list.extend(batch_features)
        coords_list.extend(batch_coords.numpy())
        names_list.extend(batch_names)

    return {
        "features_list": np.array(features_list, dtype=np.float32),
        "coords_list": np.array(coords_list, dtype=np.int32),
        "names_list": np.array(names_list, dtype=object),
    }


def save_feature(save_path: Path, data_dict: dict):
    """Aggregate flat patch lists into per-slide entries."""
    feature_dict = {}
    results = {"names_list": [], "features_list": [], "coords_list": []}

    for feature, coord, name in zip(
        data_dict["features_list"],
        data_dict["coords_list"],
        data_dict["names_list"],
    ):
        if name not in feature_dict:
            feature_dict[name] = {"features": [], "coords": []}
        feature_dict[name]["features"].append(feature)
        feature_dict[name]["coords"].append(coord)

    for name in feature_dict:
        feature_dict[name]["features"] = torch.stack([
            torch.from_numpy(f) if isinstance(f, np.ndarray) else f
            for f in feature_dict[name]["features"]
        ])
        feature_dict[name]["coords"] = torch.from_numpy(np.array(feature_dict[name]["coords"]))
        results["names_list"].append(name)
        results["features_list"].append(feature_dict[name]["features"])
        results["coords_list"].append(feature_dict[name]["coords"])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as handle:
        pickle.dump(results, handle)


def main(
    dataset_dir: Path,
    gpu: int = 0,
    assets_dir: str | Path = "assets/ckpts",
    batch_size: int = 256,
    num_workers: int = 4,
    force: bool = False,
):
    dataset_dir = Path(dataset_dir)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model, transform = load_model_and_transforms(device, assets_dir=assets_dir)

    save_path = dataset_dir / "data_features.pkl"
    if save_path.exists() and not force:
        logger.info(f"Features already exist at {save_path}; skipping (use --force to overwrite)")
        return

    patches_dir = dataset_dir / "patches"
    if not patches_dir.is_dir():
        raise FileNotFoundError(f"Patch directory not found: {patches_dir}")

    logger.info(f"Extracting UNI-2h features for {dataset_dir.name}")
    data_dict = extract_features_from_patches(
        model,
        patches_dir,
        transform,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    save_feature(save_path, data_dict)
    logger.info(f"Saved features to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract UNI-2h features for TCGA patch folders.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset root, e.g. data/TCGA-THYM (must contain patches/)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--assets-dir",
        default="assets/ckpts",
        help="Local directory for UNI-2h weights",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data_features.pkl")
    args = parser.parse_args()

    main(
        dataset_dir=Path(args.dataset_dir),
        gpu=args.gpu,
        assets_dir=args.assets_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force,
    )
