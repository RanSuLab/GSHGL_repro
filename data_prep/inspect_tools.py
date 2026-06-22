"""Inspection helpers for h5, pt, WSI, and gene coverage summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


def inspect_h5(file_path: Path) -> None:
    import h5py

    with h5py.File(file_path, "r") as handle:
        print(f"File: {file_path}")
        print(f"Top-level keys: {list(handle.keys())}")
        for key in ["annots", "features", "coords", "coords_patching"]:
            if key in handle:
                print(f"{key} shape: {handle[key].shape}")
            else:
                print(f"{key} missing")


def inspect_pt(file_path: Path) -> None:
    import torch

    data = torch.load(file_path, map_location="cpu", weights_only=False)
    print(f"File: {file_path}")
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    else:
        print(data)


def inspect_wsi_levels(svs_path: Path) -> None:
    import openslide

    slide = openslide.open_slide(str(svs_path))
    print(f"Slide: {svs_path}")
    print(f"level_dimensions = {slide.level_dimensions}")
    print(f"level_downsamples = {slide.level_downsamples}")

    mask_level = len(slide.level_dimensions) - 1
    for index, dims in enumerate(slide.level_dimensions):
        if max(dims) < 2000:
            mask_level = index
            break

    print(f"mask_level = {mask_level}")
    print(f"mask_size = {slide.level_dimensions[mask_level]}")
    cut_level = slide.get_best_level_for_downsample(4)
    print(f"cut_level = {cut_level}")
    print(f"cut_size = {slide.level_dimensions[cut_level]}")


def gene_summary(gene_root: Path, topn: int = 20) -> None:
    gene_to_variants = defaultdict(lambda: defaultdict(list))
    processed_wsis = set()
    all_wsis = set()

    for wsi_path in gene_root.iterdir():
        if not wsi_path.is_dir():
            continue
        all_wsis.add(wsi_path.name)
        for tsv_file in wsi_path.glob("*.tsv"):
            try:
                frame = pd.read_csv(tsv_file, sep="\t")
                if "Hugo_Symbol" not in frame.columns or "Variant_Classification" not in frame.columns:
                    continue
                for _, row in frame.iterrows():
                    gene_name = row["Hugo_Symbol"]
                    variant_type = row["Variant_Classification"]
                    if pd.notna(gene_name) and pd.notna(variant_type):
                        gene_to_variants[gene_name][variant_type].append(wsi_path.name)
                        processed_wsis.add(wsi_path.name)
            except Exception as exc:
                print(f"Failed to read {tsv_file}: {exc}")

    print(f"Unique genes: {len(gene_to_variants)}")
    print(f"Processed WSIs: {len(processed_wsis)}")
    print(f"WSI subdirectories: {len(all_wsis)}")

    gene_wsi_counts = []
    for gene_name, variant_dict in gene_to_variants.items():
        all_wsis_for_gene = set()
        for wsi_list in variant_dict.values():
            all_wsis_for_gene.update(set(wsi_list))
        gene_wsi_counts.append((gene_name, len(all_wsis_for_gene), variant_dict))

    gene_wsi_counts.sort(key=lambda item: item[1], reverse=True)
    top_genes = gene_wsi_counts[:topn]

    print(f"\nTop {topn} genes by WSI coverage:")
    print("-" * 80)
    for gene_name, total_wsis_for_gene, variant_dict in top_genes:
        total_variants = sum(len(wsi_list) for wsi_list in variant_dict.values())
        print(f"{gene_name}: {total_variants} variant records, {total_wsis_for_gene} WSIs")
        print(f"variant types: {', '.join(variant_dict.keys())}")
