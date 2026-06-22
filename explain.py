"""Explainability entrypoint."""

from __future__ import annotations

import argparse


def main() -> None:
    """Dispatch to heatmap or top-patches visualization."""
    parser = argparse.ArgumentParser(description="Explainability entrypoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("heatmap", help="Node/patch importance heatmap (explain.heatmap)")
    sub.add_parser("top_patches", help="WSI top-patches visualization (explain.top_patches)")

    args, _unknown = parser.parse_known_args()

    if args.cmd == "heatmap":
        from explain.heatmap import main as heatmap_main

        heatmap_main()
        return

    if args.cmd == "top_patches":
        from explain.top_patches import main as top_patches_main

        top_patches_main()
        return

    raise ValueError(f"Unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    main()
