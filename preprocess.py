"""
Graph construction entrypoint.

Applies config presets and CLI overrides, saves `preprocess_config_snapshot.json`,
and calls `data.preprocess_graphs.main()` to produce graph caches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.config import PathConfig, PreprocessConfig
from configs.runtime import apply_runtime_overrides, build_config_snapshot, save_config_snapshot
from data.preprocess_graphs import main


def cli_main() -> None:
    """Save config snapshot, then run graph preprocessing."""
    parser = argparse.ArgumentParser(description="Build WSI graph data for the main pipeline.")
    parser.add_argument("--dataset-config")
    parser.add_argument("--model-config")
    parser.add_argument("--dataset")
    parser.add_argument("--gene")
    parser.add_argument("--run-name")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--pkl-filename")
    parser.add_argument("--graph-tag")
    parser.add_argument("--max-neighbors", type=int)
    parser.add_argument("--feature-k", type=int)
    parser.add_argument("--use-hybrid-graph")
    parser.add_argument("--min-patches", type=int)
    args = parser.parse_args()

    loaded_presets = apply_runtime_overrides(args)
    path_cfg = PathConfig()
    preprocess_cfg = PreprocessConfig()
    snapshot = build_config_snapshot(
        path_cfg=path_cfg,
        preprocess_cfg=preprocess_cfg,
        metadata={"presets": loaded_presets, "entrypoint": "preprocess.py"},
    )
    save_config_snapshot(
        snapshot, Path(path_cfg.GRAPH_CACHE_DIR) / "preprocess_config_snapshot.json"
    )
    main()


if __name__ == "__main__":
    cli_main()
