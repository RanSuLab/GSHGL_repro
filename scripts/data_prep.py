"""Unified CLI for TCGA data preparation.

This script is the entry point for WSI download, MAF processing, label building,
and lightweight inspection utilities used before training fWCGM.

Typical pipeline (see README for full details):

    # 1. Query & download one WSI per case from GDC
    python scripts/data_prep.py caselevel-query \\
        --project TCGA-THYM --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt
    python scripts/data_prep.py caselevel-download \\
        --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt --out data/TCGA-THYM/wsi

    # 2. Tile WSIs (separate script: scripts/extract_patches.py)

    # 3. Build per-case gene tables from somatic MAF
    python scripts/data_prep.py maf-build --dataset-dir data/TCGA-THYM

    # 4. Build training labels (or use scripts/build_labels.py)
    python scripts/data_prep.py labels-build \\
        --feature-path data/TCGA-THYM/data_features.pkl \\
        --gene-dir data/TCGA-THYM/gene --gene-name GTF2I

Subcommands are implemented in data_prep/{wsi,maf,label,inspect}_tools.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from the repo root when invoked as `python scripts/data_prep.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# WSI commands — sync local files or fetch from GDC
# ---------------------------------------------------------------------------


def command_wsi_sync(args: argparse.Namespace) -> None:
    """Collect scattered .svs files under Dataset/ into a flat wsi/ folder."""
    from data_prep.wsi_tools import sync_dataset_to_wsi

    sync_dataset_to_wsi(args.root)


def command_caselevel_query(args: argparse.Namespace) -> None:
    """Query GDC and write a manifest with one primary WSI per TCGA case."""
    from data_prep.wsi_tools import query_caselevel_manifest

    query_caselevel_manifest(args.project, args.manifest)


def command_caselevel_download(args: argparse.Namespace) -> None:
    """Download WSIs listed in a GDC manifest (requires gdc-client)."""
    from data_prep.wsi_tools import download_from_manifest

    download_from_manifest(args.manifest, args.out)


# ---------------------------------------------------------------------------
# MAF commands — somatic mutation tables for label construction
# ---------------------------------------------------------------------------


def command_maf_build(args: argparse.Namespace) -> None:
    """Query/download MAF files and convert them to per-case gene TSV tables.

    Expects patches/ under dataset-dir (used to enumerate cases).
    Writes maf_files/ and gene/{case_id}/{case_id}.tsv.
    """
    from data_prep.maf_tools import build_maf_gene_tables

    build_maf_gene_tables(args.dataset_dir)


def command_maf_recover(args: argparse.Namespace) -> None:
    """Rebuild manifest_maf_downloaded.txt from files already on disk."""
    from data_prep.maf_tools import recover_maf_success_manifest

    recover_maf_success_manifest(args.out)


def command_maf_stats(args: argparse.Namespace) -> None:
    """Print per-case MAF file counts under the dataset output directory."""
    from data_prep.maf_tools import maf_stats_per_case

    maf_stats_per_case(args.out)


# ---------------------------------------------------------------------------
# Label command — align features with mutation status
# ---------------------------------------------------------------------------


def command_labels_build(args: argparse.Namespace) -> None:
    """Merge UNI-2h features with binary gene-mutation labels into a .pkl file.

    Slides without a recorded variant for --gene-name are labeled wild-type (0).
    """
    from data_prep.label_tools import build_label_dataset

    build_label_dataset(args.feature_path, args.gene_dir, args.gene_name, args.out)


# ---------------------------------------------------------------------------
# Inspect commands — quick checks during data preparation
# ---------------------------------------------------------------------------


def command_inspect_h5(args: argparse.Namespace) -> None:
    """Print keys, shapes, and dtypes inside an HDF5 file."""
    from data_prep.inspect_tools import inspect_h5

    inspect_h5(args.file)


def command_inspect_pt(args: argparse.Namespace) -> None:
    """Print tensor shapes and metadata inside a PyTorch .pt checkpoint."""
    from data_prep.inspect_tools import inspect_pt

    inspect_pt(args.file)


def command_inspect_wsi(args: argparse.Namespace) -> None:
    """List pyramid levels and dimensions for a single .svs slide."""
    from data_prep.inspect_tools import inspect_wsi_levels

    inspect_wsi_levels(args.svs)


def command_gene_summary(args: argparse.Namespace) -> None:
    """Summarize how many cases carry each Hugo_Symbol in gene/ TSV files."""
    from data_prep.inspect_tools import gene_summary

    gene_summary(args.gene_root, args.topn)


def build_parser() -> argparse.ArgumentParser:
    epilog = """examples:
  %(prog)s caselevel-query --project TCGA-THYM --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt
  %(prog)s caselevel-download --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt --out data/TCGA-THYM/wsi
  %(prog)s maf-build --dataset-dir data/TCGA-THYM
  %(prog)s labels-build --feature-path data/TCGA-THYM/data_features.pkl --gene-dir data/TCGA-THYM/gene --gene-name GTF2I
  %(prog)s gene-summary --gene-root data/TCGA-THYM/gene --topn 20
"""
    parser = argparse.ArgumentParser(
        description="TCGA data preparation CLI for the fWCGM reproduction pipeline.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- WSI ---
    wsi_sync = sub.add_parser(
        "wsi-sync",
        help="Collect Dataset/ into wsi/ and normalize .svs filenames",
    )
    wsi_sync.add_argument("--root", type=Path, default=Path("data"), help="Data root containing Dataset/")
    wsi_sync.set_defaults(func=command_wsi_sync)

    case_query = sub.add_parser(
        "caselevel-query",
        help="Build a GDC manifest with one primary WSI per case",
    )
    case_query.add_argument("--project", required=True, help="GDC project ID, e.g. TCGA-THYM")
    case_query.add_argument("--manifest", type=Path, required=True, help="Output manifest .txt path")
    case_query.set_defaults(func=command_caselevel_query)

    case_download = sub.add_parser(
        "caselevel-download",
        help="Download WSIs from a GDC manifest via gdc-client",
    )
    case_download.add_argument("--manifest", type=Path, required=True, help="GDC manifest .txt")
    case_download.add_argument("--out", type=Path, required=True, help="Directory for downloaded .svs files")
    case_download.set_defaults(func=command_caselevel_download)

    # --- MAF ---
    maf_build = sub.add_parser(
        "maf-build",
        help="Query/download MAF and build per-case gene/ TSV tables",
    )
    maf_build.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Dataset root (must contain patches/; writes maf_files/ and gene/)",
    )
    maf_build.set_defaults(func=command_maf_build)

    maf_recover = sub.add_parser(
        "maf-recover",
        help="Rebuild manifest_maf_downloaded.txt from existing maf_files/",
    )
    maf_recover.add_argument("--out", type=Path, required=True, help="Dataset output directory")
    maf_recover.set_defaults(func=command_maf_recover)

    maf_stats = sub.add_parser(
        "maf-stats",
        help="Summarize downloaded MAF file counts per case",
    )
    maf_stats.add_argument("--out", type=Path, required=True, help="Dataset output directory")
    maf_stats.set_defaults(func=command_maf_stats)

    # --- Labels ---
    labels = sub.add_parser(
        "labels-build",
        help="Build labels_{GENE}.pkl from features + gene TSV tables",
    )
    labels.add_argument("--feature-path", type=Path, required=True, help="data_features.pkl from UNI-2h")
    labels.add_argument("--gene-dir", type=Path, required=True, help="Directory with per-case gene TSV files")
    labels.add_argument("--gene-name", required=True, help="Hugo symbol, e.g. GTF2I")
    labels.add_argument("--out", type=Path, default=None, help="Output .pkl (default: labels_{GENE}.pkl next to features)")
    labels.set_defaults(func=command_labels_build)

    # --- Inspect ---
    inspect_h5_cmd = sub.add_parser("inspect-h5", help="Inspect structure of an HDF5 file")
    inspect_h5_cmd.add_argument("--file", type=Path, required=True)
    inspect_h5_cmd.set_defaults(func=command_inspect_h5)

    inspect_pt_cmd = sub.add_parser("inspect-pt", help="Inspect structure of a PyTorch .pt file")
    inspect_pt_cmd.add_argument("--file", type=Path, required=True)
    inspect_pt_cmd.set_defaults(func=command_inspect_pt)

    inspect_wsi_cmd = sub.add_parser("inspect-wsi", help="Inspect WSI pyramid levels for one .svs")
    inspect_wsi_cmd.add_argument("--svs", type=Path, required=True)
    inspect_wsi_cmd.set_defaults(func=command_inspect_wsi)

    gene_summary_cmd = sub.add_parser(
        "gene-summary",
        help="Summarize Hugo_Symbol coverage across gene/ TSV files",
    )
    gene_summary_cmd.add_argument("--gene-root", type=Path, required=True, help="gene/ directory from maf-build")
    gene_summary_cmd.add_argument("--topn", type=int, default=20, help="Number of top genes to print")
    gene_summary_cmd.set_defaults(func=command_gene_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
