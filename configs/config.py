"""Default configuration classes for paths, preprocessing, pretraining, and training."""

import os
from copy import deepcopy


FREE_GPU = 0


def _coerce_env_value(raw, current):
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(obj, prefix):
    for key, value in vars(obj).items():
        env_key = f"{prefix}_{key}"
        raw = os.getenv(env_key)
        if raw is not None:
            setattr(obj, key, _coerce_env_value(raw, value))
    return obj


class PathConfig:
    def __init__(self):
        self.DATASET_NAME = "ACC"
        self.GENE_NAME = "TP53"
        self.SELECTED_GPU = FREE_GPU
        self.NUM_CLASSES = 2
        self.run_name = "repro"
        self.DATA_ROOT = "./data"
        self.OUTPUT_ROOT = "output"
        self.PKL_FILENAME = None
        self.GRAPH_TAG = "hybrid"

        _apply_env_overrides(self, "FWCGM_PATH")

        default_pkl = self.PKL_FILENAME or f"labels_{self.GENE_NAME}.pkl"
        self.PKL_PATH = (
            f"{self.DATA_ROOT}/TCGA-{self.DATASET_NAME}/{default_pkl}"
        )

        base_graph = f"{self.OUTPUT_ROOT}/TCGA-{self.DATASET_NAME}_{self.GENE_NAME}/graphs"
        self.GRAPH_SLIDE_DIR = f"{base_graph}/slides"
        self.GRAPH_CACHE_DIR = f"{base_graph}/cache"
        self.GRAPH_CACHE_FILE = f"{self.GRAPH_CACHE_DIR}/cache_graphs_{self.GRAPH_TAG}_{self.GENE_NAME}.pt"

        base_run = (
            f"{self.OUTPUT_ROOT}/TCGA-{self.DATASET_NAME}_{self.GENE_NAME}/"
            f"{self.run_name}"
        )
        self.RUN_DIR = base_run
        self.PRETRAIN_DIR = f"{base_run}/pretrain"
        self.RESULTS_DIR = f"{base_run}/results"


class PreprocessConfig:
    def __init__(self):
        self.MAX_NEIGHBORS = 16
        self.FEATURE_K = 6
        self.USE_HYBRID_GRAPH = True
        self.ADAPTIVE_K = True
        self.NORMALIZE_COORDS = True
        self.NORMALIZE_FEATURES = True
        self.MIN_PATCHES = 2
        self.VERBOSE_SKIP = True
        self.EDGE_DIM = 4

        _apply_env_overrides(self, "FWCGM_PREPROCESS")


class PretrainConfig:
    DEFAULTS = {
        "INPUT_DIM": 1536,
        "HIDDEN_DIM": 512,
        "GAT_HEADS": 8,
        "DROPOUT": 0.5,
        "SEED": 42,
        "NUM_EPOCHS": 50,
        "BATCH_SIZE_PER_GPU": 1,
        "ACCUMULATION_STEPS": 2,
        "GRADIENT_CHECKPOINTING": False,
        "LEARNING_RATE": 1e-5,
        "WEIGHT_DECAY": 1e-4,
        "TEMPERATURE": 0.5,
        "DROP_NODE_P": 0.05,
        "MASK_FEAT_P": 0.05,
        "DROP_EDGE_P": 0.1,
        "PROJ_DIM": 256,
        "PRED_DIM": 256,
        "EDGE_DIM": 4,
        "AUX_CLS_WEIGHT": 0.5,
        "AUX_LABEL_SMOOTHING": 0.05,
        "USE_AUX_CLASSIFICATION": True,
    }

    def __init__(self):
        for key, value in deepcopy(self.DEFAULTS).items():
            setattr(self, key, value)
        _apply_env_overrides(self, "FWCGM_PRETRAIN")


class TrainConfig:
    DEFAULTS = {
        "INPUT_DIM": 1536,
        "HIDDEN_DIM": 512,
        "GAT_HEADS": 8,
        "DROPOUT": 0.5,
        "NUM_EPOCHS": 200,
        "BATCH_SIZE": 4,
        "ACCUMULATION_STEPS": 2,
        "LEARNING_RATE": 5e-4,
        "WEIGHT_DECAY": 1e-4,
        "LABEL_SMOOTHING": 0.1,
        "N_SPLITS": 5,
        "RANDOM_STATE": 42,
        "PATIENCE": 50,
        "SEED": 42,
        "USE_FOCAL_LOSS": False,
        "FOCAL_GAMMA": 2.0,
        "FOCAL_ALPHA": None,
        "FUSION_NUM_HEADS": 4,
        "FUSION_NUM_LAYERS": 2,
        "FUSION_DROPOUT": 0.2,
        "FUSION_TYPE": "cross_scale_transformer",
        "TOKEN_SCHEME": "patch_region",
    }

    def __init__(self):
        for key, value in deepcopy(self.DEFAULTS).items():
            setattr(self, key, value)
        _apply_env_overrides(self, "FWCGM_TRAIN")
