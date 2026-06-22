"""
Training entrypoint.

Saves `config_snapshot.json` under the run directory for reproducibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from configs.config import PathConfig, PretrainConfig, TrainConfig
from configs.runtime import apply_runtime_overrides, build_config_snapshot, save_config_snapshot
from train.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="fWCGM training entrypoint.")
    parser.add_argument("--dataset-config")
    parser.add_argument("--experiment-config")
    parser.add_argument("--model-config")
    parser.add_argument("--dataset")
    parser.add_argument("--gene")
    parser.add_argument("--run-name")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--pkl-filename")
    parser.add_argument("--graph-tag")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--splits", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--pretrain-batch-size", type=int)
    parser.add_argument("--pretrain-lr", type=float)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--pretrain-accumulation-steps", type=int)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in the encoder (trades compute for memory).",
    )
    parser.add_argument("--fusion-layers", type=int)
    parser.add_argument("--fusion-heads", type=int)
    parser.add_argument("--fusion-dropout", type=float)
    parser.add_argument("--fusion-type")
    parser.add_argument("--token-scheme")
    parser.add_argument("--aux-cls-weight", type=float)
    parser.add_argument("--use-aux-classification")
    args = parser.parse_args()

    loaded_presets = apply_runtime_overrides(args)

    path_cfg = PathConfig()
    pre_cfg = PretrainConfig()
    train_cfg = TrainConfig()
    snapshot = build_config_snapshot(
        path_cfg=path_cfg,
        pretrain_cfg=pre_cfg,
        train_cfg=train_cfg,
        metadata={"presets": loaded_presets, "entrypoint": "train.py"},
    )
    save_config_snapshot(snapshot, Path(path_cfg.RUN_DIR) / "config_snapshot.json")

    logger.info("Starting training")
    run_training()
    logger.info("Training finished")


if __name__ == "__main__":
    main()
