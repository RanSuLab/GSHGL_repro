"""Thin wrapper around data_prep.label_tools for building labels_{GENE}.pkl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_prep.label_tools import build_label_dataset


def main():
    parser = argparse.ArgumentParser(description="Build labels_{GENE}.pkl for fWCGM preprocessing.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset root, e.g. data/TCGA-THYM")
    parser.add_argument("--gene", required=True, help="Target gene symbol, e.g. GTF2I")
    parser.add_argument("--feature-path", default=None, help="Default: <dataset-dir>/data_features.pkl")
    parser.add_argument("--gene-dir", default=None, help="Default: <dataset-dir>/gene")
    parser.add_argument("--out", default=None, help="Default: <dataset-dir>/labels_{GENE}.pkl")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    feature_path = Path(args.feature_path or dataset_dir / "data_features.pkl")
    gene_dir = Path(args.gene_dir or dataset_dir / "gene")
    out_path = Path(args.out) if args.out else dataset_dir / f"labels_{args.gene}.pkl"

    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    if not gene_dir.is_dir():
        raise FileNotFoundError(f"Gene directory not found: {gene_dir}")

    build_label_dataset(feature_path, gene_dir, args.gene, out_path)


if __name__ == "__main__":
    main()
