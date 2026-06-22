"""Align UNI-2h features with per-slide gene TSV files."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd


def extract_gene_variants(gene_dir: Path, gene_name: str):
    variant_to_wsi = defaultdict(list)
    wsi_to_variant = {}
    not_compiled_wsis = []

    for tsv_file in gene_dir.rglob("*.tsv"):
        wsi_name = tsv_file.stem
        try:
            frame = pd.read_csv(tsv_file, sep="\t")
            if "Hugo_Symbol" not in frame.columns or "Variant_Classification" not in frame.columns:
                not_compiled_wsis.append(wsi_name)
                continue

            gene_rows = frame[frame["Hugo_Symbol"] == gene_name]
            if gene_rows.empty:
                not_compiled_wsis.append(wsi_name)
                continue

            variant_type = gene_rows.iloc[-1]["Variant_Classification"]
            variant_to_wsi[variant_type].append(wsi_name)
            wsi_to_variant[wsi_name] = variant_type
        except Exception:
            not_compiled_wsis.append(wsi_name)

    if not_compiled_wsis:
        variant_to_wsi["Not Compiled"] = not_compiled_wsis
        for wsi in not_compiled_wsis:
            wsi_to_variant[wsi] = "Not Compiled"

    return dict(variant_to_wsi), wsi_to_variant


def build_label_dataset(
    feature_path: Path,
    gene_dir: Path,
    gene_name: str,
    out_path: Path | None = None,
) -> Path:
    data_features = joblib.load(feature_path)
    _, wsi_to_variant = extract_gene_variants(gene_dir, gene_name)

    labels_list = []
    names_list = []
    coords_list = []
    features_list = []

    for name, coord, feature in zip(
        data_features["names_list"],
        data_features["coords_list"],
        data_features["features_list"],
    ):
        if name not in wsi_to_variant:
            continue
        labels_list.append(0 if wsi_to_variant[name] == "Not Compiled" else 1)
        names_list.append(name)
        coords_list.append(coord)
        features_list.append(feature)

    prepared_data = {
        "names_list": names_list,
        "coords_list": coords_list,
        "features_list": features_list,
        "labels_list": labels_list,
    }

    if out_path is None:
        out_path = gene_dir.parent / f"labels_{gene_name}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(prepared_data, out_path)
    print(f"Saved {out_path} with {len(labels_list)} slides")
    return out_path
