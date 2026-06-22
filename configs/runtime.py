"""Runtime config loading: presets, CLI overrides, and config snapshots."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

SECTION_ENV_PREFIX = {
    "path": "FWCGM_PATH",
    "preprocess": "FWCGM_PREPROCESS",
    "pretrain": "FWCGM_PRETRAIN",
    "train": "FWCGM_TRAIN",
}

CLI_ENV_MAP = {
    "dataset": "FWCGM_PATH_DATASET_NAME",
    "gene": "FWCGM_PATH_GENE_NAME",
    "gpu": "FWCGM_PATH_SELECTED_GPU",
    "run_name": "FWCGM_PATH_run_name",
    "data_root": "FWCGM_PATH_DATA_ROOT",
    "output_root": "FWCGM_PATH_OUTPUT_ROOT",
    "pkl_filename": "FWCGM_PATH_PKL_FILENAME",
    "graph_tag": "FWCGM_PATH_GRAPH_TAG",
    "epochs": "FWCGM_TRAIN_NUM_EPOCHS",
    "batch_size": "FWCGM_TRAIN_BATCH_SIZE",
    "splits": "FWCGM_TRAIN_N_SPLITS",
    "patience": "FWCGM_TRAIN_PATIENCE",
    "lr": "FWCGM_TRAIN_LEARNING_RATE",
    "pretrain_epochs": "FWCGM_PRETRAIN_NUM_EPOCHS",
    "pretrain_batch_size": "FWCGM_PRETRAIN_BATCH_SIZE_PER_GPU",
    "pretrain_lr": "FWCGM_PRETRAIN_LEARNING_RATE",
    "accumulation_steps": "FWCGM_TRAIN_ACCUMULATION_STEPS",
    "pretrain_accumulation_steps": "FWCGM_PRETRAIN_ACCUMULATION_STEPS",
    "max_neighbors": "FWCGM_PREPROCESS_MAX_NEIGHBORS",
    "feature_k": "FWCGM_PREPROCESS_FEATURE_K",
    "use_hybrid_graph": "FWCGM_PREPROCESS_USE_HYBRID_GRAPH",
    "min_patches": "FWCGM_PREPROCESS_MIN_PATCHES",
    "fusion_layers": "FWCGM_TRAIN_FUSION_NUM_LAYERS",
    "fusion_heads": "FWCGM_TRAIN_FUSION_NUM_HEADS",
    "fusion_dropout": "FWCGM_TRAIN_FUSION_DROPOUT",
    "fusion_type": "FWCGM_TRAIN_FUSION_TYPE",
    "token_scheme": "FWCGM_TRAIN_TOKEN_SCHEME",
    "aux_cls_weight": "FWCGM_PRETRAIN_AUX_CLS_WEIGHT",
    "use_aux_classification": "FWCGM_PRETRAIN_USE_AUX_CLASSIFICATION",
}

PRESET_ARG_MAP = {
    "dataset_config": "dataset",
    "experiment_config": "experiment",
    "model_config": "model",
}


def _normalize_value(value):
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _set_env(section, key, value):
    os.environ[f"{SECTION_ENV_PREFIX[section]}_{key}"] = str(_normalize_value(value))


def apply_section_overrides(section, values):
    for key, value in values.items():
        if value is not None:
            _set_env(section, key, value)


def _preset_path(kind, value):
    candidate = Path(value)
    if candidate.exists():
        return candidate
    suffix = ".json" if candidate.suffix == "" else ""
    return CONFIG_DIR / kind / f"{value}{suffix}"


def load_preset_file(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return payload


def apply_preset_overrides(*, dataset_config=None, experiment_config=None, model_config=None):
    preset_names = {
        "dataset": dataset_config,
        "experiment": experiment_config,
        "model": model_config,
    }
    loaded = {}
    for kind, value in preset_names.items():
        if not value:
            continue
        path = _preset_path(kind, value)
        payload = load_preset_file(path)
        loaded[kind] = {"path": str(path), "payload": payload}
        for section, section_values in payload.items():
            if section not in SECTION_ENV_PREFIX:
                raise ValueError(f"Unknown config section {section} in {path}")
            if not isinstance(section_values, dict):
                raise ValueError(f"Config section {section} must be an object: {path}")
            apply_section_overrides(section, section_values)
    return loaded


def apply_runtime_overrides(args):
    loaded = apply_preset_overrides(
        dataset_config=getattr(args, "dataset_config", None),
        experiment_config=getattr(args, "experiment_config", None),
        model_config=getattr(args, "model_config", None),
    )

    for attr, env_key in CLI_ENV_MAP.items():
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_key] = str(_normalize_value(value))

    if getattr(args, "gradient_checkpointing", False):
        _set_env("pretrain", "GRADIENT_CHECKPOINTING", True)

    return loaded


def build_config_snapshot(*, path_cfg, preprocess_cfg=None, pretrain_cfg=None, train_cfg=None, metadata=None):
    snapshot = {
        "path": dict(vars(path_cfg)),
    }
    if preprocess_cfg is not None:
        snapshot["preprocess"] = dict(vars(preprocess_cfg))
    if pretrain_cfg is not None:
        snapshot["pretrain"] = dict(vars(pretrain_cfg))
    if train_cfg is not None:
        snapshot["train"] = dict(vars(train_cfg))
    if metadata:
        snapshot["metadata"] = metadata
    return snapshot


def save_config_snapshot(snapshot, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def env_for_experiment(
    dataset,
    gene,
    run_name,
    *,
    gpu=None,
    data_root=None,
    output_root=None,
    dataset_config=None,
    experiment_config=None,
    model_config=None,
):
    env = os.environ.copy()
    env["FWCGM_PATH_DATASET_NAME"] = str(dataset)
    env["FWCGM_PATH_GENE_NAME"] = str(gene)
    env["FWCGM_PATH_run_name"] = str(run_name)

    if gpu is not None:
        env["FWCGM_PATH_SELECTED_GPU"] = str(gpu)
    if data_root is not None:
        env["FWCGM_PATH_DATA_ROOT"] = str(data_root)
    if output_root is not None:
        env["FWCGM_PATH_OUTPUT_ROOT"] = str(output_root)
    if dataset_config is not None:
        env["FWCGM_DATASET_CONFIG"] = str(dataset_config)
    if experiment_config is not None:
        env["FWCGM_EXPERIMENT_CONFIG"] = str(experiment_config)
    if model_config is not None:
        env["FWCGM_MODEL_CONFIG"] = str(model_config)

    return env
