# -*- coding: utf-8 -*-
"""
Feature-Profile Enhanced Learning (FPEL) benchmark script.

**FPEL** layers **feature-profile soft partitions** and **region-wise** experts of the same
interpretable base-learner type on top of a **white-box** pipeline, improving local adaptation under
heterogeneous regimes while keeping a transparent regional structure. Saved tables still use
``Structure=ProFPEL`` and the ``ProFPELModel`` class name for continuity with earlier exports.

Empirical claim (conditional, not universal): for the same **white-box** base learner B,
**FPEL-B** (``Structure`` = ``ProFPEL``) can improve or match **Pure-B** on held-out data when
response structure varies across **feature-profile** regimes; gains are neither guaranteed nor uniform.

**Black-box baselines** (RandomForest, XGBoost, LightGBM, CatBoost, MLP) are **standalone**
tuned predictors: train-only inner cross-validation on ``BLACK_BOX_TUNING_GRIDS`` (tree depth,
learning rate, leaf or estimator counts, and related axes; a compact default grid, wider when
``FPPEL_FULL_GRID`` is set). They **benchmark** how close **FPEL white-box** models come to
strong opaque predictors on the same tasks.

Main **Pure-B vs FPEL-B** matrix (same inner-CV grid per white-box B):

1. Linear / sparse linear: Ridge-L, Ridge-Q, ElasticNet-L, ElasticNet-Q
2. Additive / smooth: Spline-Ridge, GAM, EBM (main effects), GA2M (Interpret EBM with interactions; optional interpret)
3. Tree / rule (interpretable shallow rules): DecisionTree

Core protocol:
  - Biomass gasification: real experimental data, stratified by bed material (**only benchmark dataset** in this script).
  - Hyperparameter selection: train-only inner CV / OOF protocol.
  - Task scope: **regression** in this version; classification adapters remain as
    future-extension scaffolding alongside the reported regression protocol.
  - Main comparison: **Pure-B vs FPEL-B** (``Structure`` uses ``Pure`` or ``ProFPEL``).
    Base-learner hyperparameters use the **same** inner-CV grid and split construction for both
    structures. **Pure-B** selects them on the Pure model. **FPEL-B** repeats the **same**
    Pure-equivalent white-box inner CV, fixes the winner, then runs **fold-local static FPEL**
    fits for stacked OOF and a full-training refit (no outer loop over partition gates).
    **Train_Time** is measured per structure over that structure's full train+selection block so
    Pure vs FPEL timing deltas stay interpretable.
  - **Black-box baselines** (``Structure=BlackBoxBaseline``): RF, XGBoost, LightGBM, CatBoost, MLP
    use train-only inner CV on ``BLACK_BOX_TUNING_GRIDS`` as standalone predictors.
  - **Train_Time** (seconds): wall-clock time for the train+selection block of each ``Pure`` or
    ``ProFPEL`` run; paired summaries report Pure vs FPEL means and deltas.
  - **ProFPEL_LocalWeightsAccepted**: ``1`` when every final local expert ``fit`` receives
    ``sample_weight``; otherwise ``0`` with a stderr diagnostic listing experts that trained without weights.
  - **PROFPEL_STRICT_SAMPLE_WEIGHT** (alias **FPPEL_STRICT_SAMPLE_WEIGHT**): when set, require
    ``sample_weight`` support in every expert ``fit``; missing support stops the run.
  - Default auto-**K**: adaptive Gaussian-mixture **BIC** search in feature-profile space
    (``k_selection=adaptive_gmm_bic``): **K** starts at 1, increases while BIC improves, stops after a
    short stall of non-improving **K**; the chosen **K** is the best BIC seen. **K** is capped by
    ``min(n - 1, floor(n / p))`` with ``n`` the training sample size and ``p`` the input feature count.
    Explicit ``n_regions`` pins **K** for fixed-region runs.
  - Primary model: **FPEL-Ridge-Q**.
  - Ablations contrast adaptive-**K** FPEL with ``global_only`` (**K=1**), profile information
    variants, soft vs hard gates, and feature ordering on the same training folds.
  - Supervised acceptance for the final white-box estimator (``FPPEL_SUPERVISED_FALLBACK_TOL`` or
    ``select_profpel(..., supervised_fallback_tol=...)``): when enabled, if stacked fold-local FPEL OOF **R2**
    is at most white-box inner-CV **R2** plus the margin, the returned model is **Pure-B** on the locked
    hyperparameters and the full-data FPEL refit is omitted; when the margin is not configured, the
    full-data FPEL refit always runs after the fold OOF stage.

Strict sample_weight checks honor both ``PROFPEL_STRICT_SAMPLE_WEIGHT`` and ``FPPEL_STRICT_SAMPLE_WEIGHT``.

Outputs are written under:
    results_fpel/
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Limit joblib loky physical-core discovery to reduce noisy startup under tight ulimits.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGBM = True
except Exception:
    lgb = None
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    HAS_CATBOOST = True
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None
    HAS_CATBOOST = False

try:
    from interpret.glassbox import ExplainableBoostingRegressor

    HAS_INTERPRET_EBM = True
except Exception:
    ExplainableBoostingRegressor = None
    HAS_INTERPRET_EBM = False

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY_STATS = True
except Exception:
    scipy_stats = None
    HAS_SCIPY_STATS = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(SCRIPT_DIR, "results_fpel")
FIG_DIR = os.path.join(OUT_DIR, "figures")
RANDOM_STATE = 42
QUICK_MODE = (
    os.environ.get("FPPEL_QUICK", "")
    .strip()
    .lower()
    in ("1", "true", "yes", "y")
)
STRICT_SAMPLE_WEIGHT = (
    os.environ.get("PROFPEL_STRICT_SAMPLE_WEIGHT", os.environ.get("FPPEL_STRICT_SAMPLE_WEIGHT", ""))
    .strip()
    .lower()
    in ("1", "true", "yes", "y")
)
_CPU_COUNT = os.cpu_count() or 1
PARALLEL_N_JOBS = int(os.environ.get("FPPEL_N_JOBS", str(min(6, _CPU_COUNT))))
PARALLEL_BACKEND = os.environ.get("FPPEL_PARALLEL_BACKEND", "threading").strip() or "threading"
FULL_GRID = (
    os.environ.get("FPPEL_FULL_GRID", "")
    .strip()
    .lower()
    in ("1", "true", "yes", "y")
)
# Consecutive GMM-BIC steps without a new global minimum before adaptive K search stops.
ADAPTIVE_GMM_BIC_STALL_MAX = 3


def _resolve_supervised_fallback_tol(explicit: Optional[float]) -> Optional[float]:
    """
    Margin for supervised acceptance in ``select_profpel``.

    Returns ``None`` when no margin is configured (empty ``FPPEL_SUPERVISED_FALLBACK_TOL`` and
    ``explicit is None``), or when the environment value is negative: the pipeline runs a full-data
    FPEL refit after fold OOF.

    Returns a non-negative float when configured: if stacked fold-local FPEL OOF **R2** is at most
    white-box inner-CV **R2** plus that margin, the returned estimator is **Pure-B** on the locked
    hyperparameters and the full-data FPEL refit is omitted.
    """
    if explicit is not None:
        v = float(explicit)
        return None if v < 0 else v
    raw = os.environ.get("FPPEL_SUPERVISED_FALLBACK_TOL", "").strip()
    if not raw:
        return None
    v_env = float(raw)
    return None if v_env < 0 else v_env
# Subset of ``WHITEBOX_PROFPEL_LEARNERS`` when FPPEL_LEARNER_SET is legacy_core, lean, or old_core.
LEGACY_CORE_MAIN_BASE_LEARNERS = (
    "ridge_quadratic",
    "spline_ridge",
    "gam",
    "decision_tree",
)
TYPED_MAIN_BASE_LEARNERS = (
    "ridge_quadratic",
    "elasticnet_quadratic",
    "spline_ridge",
    "gam",
    "ebm",
    "ga2m",
    "decision_tree",
)

OUTER_SPLITS = int(os.environ.get("FPPEL_OUTER_SPLITS", str(2 if QUICK_MODE else 5)))
INNER_CV_FOLDS = int(os.environ.get("FPPEL_INNER_CV", str(3 if QUICK_MODE else 5)))
BIOMASS_TEST_SIZE = 0.20

PRIMARY_BASE_LEARNER = "ridge_quadratic"

WHITEBOX_PROFPEL_LEARNERS: Tuple[str, ...] = (
    "ridge_linear",
    "ridge_quadratic",
    "elasticnet_linear",
    "elasticnet_quadratic",
    "spline_ridge",
    "gam",
    "ebm",
    "ga2m",
    "decision_tree",
)
BLACK_BOX_BASELINE_LEARNERS: Tuple[str, ...] = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
)
ALL_REPORTED_LEARNERS: Tuple[str, ...] = WHITEBOX_PROFPEL_LEARNERS + BLACK_BOX_BASELINE_LEARNERS

DEFAULT_PROFPEL_KWARGS: Dict[str, Any] = {
    "feature_order": "target_corr",
    "profile_metric": "value_slope",
    "slope_weight": 1.0,
    "inner_cv": INNER_CV_FOLDS,
    # n_regions: None (default) -> adaptive GMM-BIC on feature-profile P.
    # Explicit int fixes K (e.g. global_only / fixed-K ablations).
    "n_regions": None,
    "k_selection": "adaptive_gmm_bic",
    "k_min": 1,
    "k_max": None,
    # See ProFPELModel._resolve_k_max: rationale for capping K by min(n-1, floor(n/p)) with p=input dim.
    "min_region_samples": None,
    "bandwidth": "auto",
    "partition_mode": "soft",
    "init_strategy": "profile_kmeans",
}

# White-box grids stay moderate for inner CV. Black-box baselines use ``BLACK_BOX_TUNING_GRIDS``
# (wider axes than white-box grids); set FPPEL_FULL_GRID=1 to expand.
MODEL_FAMILY_ORDER: Tuple[str, ...] = (
    "linear_sparse",
    "additive",
    "kernel",
    "tree",
    "boosting",
    "neural",
)

BLACK_BOX_BASELINE_LABELS: Dict[str, str] = {
    "random_forest": "Tuned RandomForest (black-box baseline)",
    "xgboost": "Tuned XGBoost (black-box baseline)",
    "lightgbm": "Tuned LightGBM (black-box baseline)",
    "catboost": "Tuned CatBoost (black-box baseline)",
    "mlp": "Tuned MLP (black-box baseline)",
}

# Tree depth, learning rate, leaf size, and estimator-count axes per library (RF, XGB, LGBM, CatBoost, MLP).
BLACK_BOX_TUNING_GRIDS: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "random_forest": {
        "n_estimators": ((200, 400) if QUICK_MODE else ((400, 800, 1200, 2000) if FULL_GRID else (400, 800, 1200))),
        "max_depth": ((None, 10) if QUICK_MODE else ((None, 8, 12, 20) if FULL_GRID else (None, 10, 20))),
        "min_samples_leaf": ((2,) if QUICK_MODE else ((1, 2, 4, 8) if FULL_GRID else (1, 2, 4))),
    },
    "xgboost": {
        "n_estimators": ((400,) if QUICK_MODE else ((800, 1500, 2500) if FULL_GRID else (600, 1200))),
        "max_depth": ((3, 6) if QUICK_MODE else ((3, 4, 6) if FULL_GRID else (3, 6))),
        "learning_rate": ((0.05,) if QUICK_MODE else (0.03, 0.05)),
    },
    "lightgbm": {
        "n_estimators": ((400,) if QUICK_MODE else ((800, 1500) if FULL_GRID else (600, 1200))),
        "num_leaves": ((15, 31) if QUICK_MODE else ((15, 31, 63) if FULL_GRID else (15, 31, 63))),
        "learning_rate": ((0.05,) if QUICK_MODE else (0.03, 0.05)),
    },
    "catboost": {
        "iterations": ((400,) if QUICK_MODE else ((800, 1500) if FULL_GRID else (600, 1200))),
        "depth": ((4, 6) if QUICK_MODE else ((4, 6, 8) if FULL_GRID else (4, 6))),
        "learning_rate": ((0.05,) if QUICK_MODE else (0.03, 0.05)),
    },
    "mlp": {
        "hidden_layer_sizes": (((64, 32),) if QUICK_MODE else ((64, 32), (128, 64))),
        "alpha": ((1e-3,) if QUICK_MODE else (1e-4, 1e-3)),
        "max_iter": ((600,) if QUICK_MODE else ((800, 1200) if FULL_GRID else (800,))),
    },
}


BASE_LEARNER_SPECS: Dict[str, Dict[str, Any]] = {
    "ridge_linear": {
        "short": "Ridge-L",
        "pure_label": "Pure Ridge-L",
        "cv_label": "FPEL-Ridge-L",
        "group": "main",
        "model_family": "linear_sparse",
        "model_family_label": "Linear / sparse linear",
        "grid": {"alpha": (0.1, 1.0, 10.0)},
    },
    "ridge_quadratic": {
        "short": "Ridge-Q",
        "pure_label": "Pure Ridge-Q",
        "cv_label": "FPEL-Ridge-Q",
        "group": "main",
        "model_family": "linear_sparse",
        "model_family_label": "Linear / sparse linear",
        "grid": {"alpha": (0.1, 1.0, 10.0)},
    },
    "elasticnet_linear": {
        "short": "ElasticNet-L",
        "pure_label": "Pure ElasticNet-L",
        "cv_label": "FPEL-ElasticNet-L",
        "group": "main",
        "model_family": "linear_sparse",
        "model_family_label": "Linear / sparse linear",
        "grid": {
            "alpha": (0.001, 0.01, 0.1),
            "l1_ratio": (0.15, 0.50, 0.85),
        },
    },
    "elasticnet_quadratic": {
        "short": "ElasticNet-Q",
        "pure_label": "Pure ElasticNet-Q",
        "cv_label": "FPEL-ElasticNet-Q",
        "group": "main",
        "model_family": "linear_sparse",
        "model_family_label": "Linear / sparse linear",
        "grid": {
            "alpha": (0.001, 0.01, 0.1),
            "l1_ratio": (0.15, 0.50, 0.85),
        },
    },
    "spline_ridge": {
        "short": "Spline-Ridge",
        "pure_label": "Pure Spline-Ridge",
        "cv_label": "FPEL-Spline-Ridge",
        "group": "main",
        "model_family": "additive",
        "model_family_label": "Additive / smooth",
        "grid": {
            "alpha": (0.1, 1.0, 10.0),
            "n_knots": (4, 5) if QUICK_MODE or FULL_GRID else (5,),
        },
    },
    "gam": {
        "short": "GAM",
        "pure_label": "Pure GAM",
        "cv_label": "FPEL-GAM",
        "group": "main",
        "model_family": "additive",
        "model_family_label": "Additive / smooth",
        "grid": {
            "alpha": (0.1, 1.0, 10.0) if FULL_GRID else (1.0,),
            "n_knots": (4, 5) if QUICK_MODE or FULL_GRID else (5,),
        },
    },
    "ebm": {
        "short": "EBM",
        "pure_label": "Pure EBM",
        "cv_label": "FPEL-EBM",
        "group": "main",
        "model_family": "additive",
        "model_family_label": "Additive / smooth",
        "optional_dependency": "interpret",
        "grid": {
            "max_bins": (128,) if QUICK_MODE or not FULL_GRID else (128, 256),
            "interactions": (0,),
            "smoothing_rounds": (64,) if QUICK_MODE or not FULL_GRID else (128, 256),
            "outer_bags": (1,) if QUICK_MODE or not FULL_GRID else (2,),
        },
    },
    "ga2m": {
        "short": "GA2M",
        "pure_label": "Pure GA2M",
        "cv_label": "FPEL-GA2M",
        "group": "main",
        "model_family": "additive",
        "model_family_label": "Additive / smooth",
        "optional_dependency": "interpret",
        "grid": {
            "max_bins": (128,) if QUICK_MODE or not FULL_GRID else (128, 256),
            "interactions": (10,) if QUICK_MODE or not FULL_GRID else (10, 15),
            "smoothing_rounds": (64,) if QUICK_MODE or not FULL_GRID else (128, 256),
            "outer_bags": (1,) if QUICK_MODE or not FULL_GRID else (2,),
        },
    },
    "decision_tree": {
        "short": "DecisionTree",
        "pure_label": "Pure DecisionTree",
        "cv_label": "FPEL-DecisionTree",
        "group": "main",
        "model_family": "tree",
        "model_family_label": "Tree / rule",
        "grid": {
            "max_depth": (2, 3, 4),
            "min_samples_leaf": (4, 8),
        },
    },
    "tree_shallow": {
        "short": "Tree",
        "pure_label": "Pure shallow tree",
        "cv_label": "FPEL-Tree",
        "group": "legacy",
        "model_family": "tree",
        "model_family_label": "Tree / rule",
        "grid": {
            "max_depth": (2, 3, 4),
            "min_samples_leaf": (4, 8),
        },
    },
    "random_forest": {
        "short": "RandomForest",
        "pure_label": "Pure RandomForest",
        "cv_label": "RandomForest (external tuned baseline)",
        "group": "main",
        "model_family": "tree",
        "model_family_label": "Tree / rule",
        "grid": {
            "n_estimators": (40,) if QUICK_MODE else (80,),
            "max_depth": (6,),
            "min_samples_leaf": (3,),
        },
    },
    "xgboost": {
        "short": "XGBoost",
        "pure_label": "Pure XGBoost",
        "cv_label": "XGBoost (external tuned baseline)",
        "group": "main",
        "model_family": "boosting",
        "model_family_label": "Boosting",
        "optional_dependency": "xgboost",
        "grid": {
            "n_estimators": (60,) if QUICK_MODE else (120,),
            "learning_rate": (0.05,),
            "max_depth": (3,),
        },
    },
    "lightgbm": {
        "short": "LightGBM",
        "pure_label": "Pure LightGBM",
        "cv_label": "LightGBM (external tuned baseline)",
        "group": "main",
        "model_family": "boosting",
        "model_family_label": "Boosting",
        "optional_dependency": "lightgbm",
        "grid": {
            "n_estimators": (60,) if QUICK_MODE else (120,),
            "learning_rate": (0.05,),
            "num_leaves": (15,),
        },
    },
    "catboost": {
        "short": "CatBoost",
        "pure_label": "Pure CatBoost",
        "cv_label": "CatBoost (external tuned baseline)",
        "group": "main",
        "model_family": "boosting",
        "model_family_label": "Boosting",
        "optional_dependency": "catboost",
        "grid": {
            "iterations": (60,) if QUICK_MODE else (120,),
            "learning_rate": (0.05,),
            "depth": (4,),
        },
    },
    "mlp": {
        "short": "MLP",
        "pure_label": "Pure MLP",
        "cv_label": "MLP (external tuned baseline)",
        "group": "main",
        "model_family": "neural",
        "model_family_label": "Neural / black-box",
        "grid": {
            "hidden_layer_sizes": ((32,),) if QUICK_MODE or not FULL_GRID else ((32,), (64, 32)),
            "alpha": (0.0001, 0.001) if FULL_GRID else (0.001,),
            "learning_rate_init": (0.001,),
        },
    },
    "logistic_regression": {
        "short": "LogisticRegression",
        "pure_label": "Pure LogisticRegression",
        "cv_label": "FPEL-LogisticRegression",
        "group": "classification",
        "model_family": "linear_sparse",
        "model_family_label": "Classification (linear)",
        "task_type": "classification",
        "grid": {"C": (0.1, 1.0, 10.0)},
    },
}

def _fpel_eprint(msg: str) -> None:
    print(f"[FPEL] {msg}", file=sys.stderr, flush=True)


def collect_runtime_environment_lines() -> List[str]:
    import sklearn

    lines = [
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"sklearn={sklearn.__version__}",
        f"matplotlib={'installed' if HAS_MATPLOTLIB else 'unavailable'}",
    ]
    try:
        import joblib

        lines.append(f"joblib={joblib.__version__}")
    except Exception:
        lines.append("joblib=unknown")
    lines.append(f"HAS_XGB={HAS_XGB}")
    lines.append(f"HAS_LGBM={HAS_LGBM}")
    lines.append(f"HAS_CATBOOST={HAS_CATBOOST}")
    lines.append(f"HAS_INTERPRET_EBM={HAS_INTERPRET_EBM}")
    lines.append(f"HAS_SCIPY_STATS={HAS_SCIPY_STATS}")
    if HAS_XGB and xgb is not None:
        lines.append(f"xgboost={getattr(xgb, '__version__', 'unknown')}")
    if HAS_LGBM and lgb is not None:
        lines.append(f"lightgbm={getattr(lgb, '__version__', 'unknown')}")
    if HAS_CATBOOST:
        try:
            import catboost as cb_mod

            lines.append(f"catboost={getattr(cb_mod, '__version__', 'unknown')}")
        except Exception:
            lines.append("catboost=unknown")
    if HAS_INTERPRET_EBM:
        try:
            import interpret as interpret_mod

            lines.append(f"interpret={getattr(interpret_mod, '__version__', 'unknown')}")
        except Exception:
            lines.append("interpret=unknown")
    lines.append(f"RANDOM_STATE={RANDOM_STATE}")
    lines.append(f"QUICK_MODE={QUICK_MODE}")
    lines.append(f"OUTER_SPLITS={OUTER_SPLITS}")
    lines.append(f"INNER_CV_FOLDS={INNER_CV_FOLDS}")
    lines.append(f"PROFPEL_STRICT_SAMPLE_WEIGHT / FPPEL_STRICT_SAMPLE_WEIGHT={STRICT_SAMPLE_WEIGHT}")
    lines.append(f"FPPEL_N_JOBS={PARALLEL_N_JOBS}")
    lines.append(f"FPPEL_PARALLEL_BACKEND={PARALLEL_BACKEND}")
    lines.append(f"FPPEL_FULL_GRID={FULL_GRID}")
    lines.append(f"FPPEL_LEARNER_SET={os.environ.get('FPPEL_LEARNER_SET', 'core')}")
    _sfb = os.environ.get("FPPEL_SUPERVISED_FALLBACK_TOL", "").strip()
    lines.append(f"FPPEL_SUPERVISED_FALLBACK_TOL={'<unset>' if not _sfb else _sfb}")
    return lines


def is_base_learner_available(learner_key: str) -> bool:
    if learner_key == "xgboost":
        return HAS_XGB
    if learner_key == "lightgbm":
        return HAS_LGBM
    if learner_key == "catboost":
        return HAS_CATBOOST
    spec = BASE_LEARNER_SPECS.get(learner_key, {})
    if spec.get("optional_dependency") == "interpret":
        return HAS_INTERPRET_EBM
    return True


def active_black_box_baselines() -> Tuple[str, ...]:
    return tuple(k for k in BLACK_BOX_BASELINE_LEARNERS if is_base_learner_available(k))


def active_main_base_learners() -> Tuple[str, ...]:
    mode = os.environ.get("FPPEL_LEARNER_SET", "core").strip().lower()
    if mode in ("typed", "representative", "balanced"):
        pool = TYPED_MAIN_BASE_LEARNERS
    elif mode in MODEL_FAMILY_ORDER:
        pool = tuple(
            k for k in WHITEBOX_PROFPEL_LEARNERS if BASE_LEARNER_SPECS[k].get("model_family") == mode
        )
    elif mode in ("legacy_core", "lean", "old_core"):
        pool = LEGACY_CORE_MAIN_BASE_LEARNERS
    else:
        pool = WHITEBOX_PROFPEL_LEARNERS
    return tuple(k for k in pool if is_base_learner_available(k))


BIOMASS_WORKBOOK_NAME = "Data of biomass gasification.xlsx"
BIOMASS_WORKBOOK_CANDIDATES = (
    os.environ.get("FPPEL_BIOMASS_PATH", ""),
    os.path.join(SCRIPT_DIR, BIOMASS_WORKBOOK_NAME),
    os.path.join(WORKSPACE_DIR, BIOMASS_WORKBOOK_NAME),
)
HHV_WORKBOOK_NAME = "HHV.xlsx"
HHV_WORKBOOK_CANDIDATES = (
    os.environ.get("FPPEL_HHV_PATH", ""),
    os.path.join(SCRIPT_DIR, HHV_WORKBOOK_NAME),
    os.path.join(WORKSPACE_DIR, HHV_WORKBOOK_NAME),
)
CO_GAS_WORKBOOK_NAME = "Co-gasification.xlsx"
CO_GAS_WORKBOOK_CANDIDATES = (
    os.environ.get("FPPEL_CO_GAS_PATH", ""),
    os.path.join(SCRIPT_DIR, CO_GAS_WORKBOOK_NAME),
    os.path.join(WORKSPACE_DIR, CO_GAS_WORKBOOK_NAME),
)

SHEET_PERCENT = "data of %"
SHEET_GY = "data of GY"
BED_COL = "Bed material"
BED_LABEL_COL = "__bed_label__"
BED_MATERIAL_VALUES = ("1", "2", "3", "4")
CONTINUOUS_COLS = ["T", "ER", "Steam/Biomass", "C", "H", "O", "Ash", "Moisture"]
BED_DUMMIES = [f"Bed_{v}" for v in BED_MATERIAL_VALUES]
FEATURE_COLUMNS = CONTINUOUS_COLS + BED_DUMMIES

TARGETS_MAP = {
    "H2": "H2 [%vol N2 free]",
    "CO": "CO [%vol N2 free]",
    "CO2": "CO2 [%vol N2 free]",
    "CH4": "CH4 [%vol N2 free]",
    "GY": "GY [Nm3/kg daf]",
}
TARGETS = list(TARGETS_MAP.keys())

HHV_FEATURE_COLUMNS = ["Ash (dry)", "C%", "H%", "O%", "N%", "S%"]
HHV_TARGETS = ["HHV"]

CO_GAS_FEATURE_COLUMNS = [
    "Biomass_C",
    "Biomass_H",
    "Biomass_N",
    "Biomass_O",
    "Biomass_S",
    "Biomass_VM",
    "Biomass_FC",
    "Biomass_Ash",
    "Coal_C",
    "Coal_H",
    "Coal_N",
    "Coal_O",
    "Coal_S",
    "Coal_VM",
    "Coal_FC",
    "Coal_Ash",
    "Blend_ratio",
    "Temperature_C",
    "Equivalent_ratio",
    "Agent_CO2",
    "Agent_Steam",
    "Agent_Air",
    "Agent_N2",
    "Agent_O2",
]
CO_GAS_TARGETS = [
    "Syngas_yield",
    "H2",
    "CO2",
    "CH4",
    "CO",
    "Syngas_LHV",
]
CO_GAS_COLUMNS = CO_GAS_FEATURE_COLUMNS + CO_GAS_TARGETS + ["Reference_DOI"]

_COL_CANDIDATES = {
    "T": ["(x1)T [oC]", "(x1)T", "T"],
    "ER": ["(x2)ER [-]", "ER"],
    "Steam/Biomass": ["(x3)Steam/Biomass", "Steam/Biomass", "S/B"],
    "C": ["(x4)C [%wt db]", "C"],
    "H": ["(x5)H [%wt db]", "H"],
    "O": ["(x6)O [%wt db]", "O"],
    "Ash": ["(x7)Ash [%wt db]", "Ash"],
    "Moisture": ["(x8)Moisture [%wt]", "Moisture"],
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_name(text: Any) -> str:
    out = "".join(ch if str(ch).isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    return out.strip("_") or "item"


def json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, default=str)


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(r"\.{2,}", ".", regex=True)
    cleaned = cleaned.where(series.notna(), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_inner_cv_splits(
    n_samples: int,
    random_state: int,
    strata: Optional[Sequence[Any]] = None,
    n_splits: int = INNER_CV_FOLDS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n_samples)
    if strata is not None:
        strata_arr = np.asarray(strata)
        if np.unique(strata_arr).size >= 2:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            return list(skf.split(idx, strata_arr))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(kf.split(idx))


def cartesian_grid(grid: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    items = list(grid.items())
    if not items:
        return [{}]
    keys = [k for k, _ in items]
    vals = [tuple(v) for _, v in items]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def score_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def weighted_r2(y: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= 1e-12:
        return float("nan")
    w = w / s
    y_mean = float(np.sum(w * y))
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def weighted_mean(values: np.ndarray, w: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= 1e-12:
        return float("nan")
    return float(np.sum(values * w) / s)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class GAMAdditiveRidgeRegressor:
    """
    Sklearn-native additive smooth model: independent spline bases per feature + global Ridge.
    Serves as a transparent GAM-style baseline without extra dependencies.
    """

    def __init__(
        self,
        n_knots: int = 5,
        alpha: float = 1.0,
        degree: int = 3,
        random_state: Optional[int] = None,
    ):
        self.n_knots = int(n_knots)
        self.alpha = float(alpha)
        self.degree = int(degree)
        self.random_state = random_state
        self.pipe_: Optional[Pipeline] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "GAMAdditiveRidgeRegressor":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        transformers = [
            (
                f"sp_{j}",
                SplineTransformer(
                    n_knots=self.n_knots,
                    degree=self.degree,
                    include_bias=False,
                    extrapolation="linear",
                ),
                [j],
            )
            for j in range(n_features)
        ]
        features = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
        ridge = Ridge(alpha=self.alpha)
        self.pipe_ = Pipeline([("features", features), ("ridge", ridge)])
        fit_kw: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kw["ridge__sample_weight"] = sample_weight
        self.pipe_.fit(X, y, **fit_kw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipe_ is None:
            raise RuntimeError("GAMAdditiveRidgeRegressor is not fitted.")
        return np.asarray(self.pipe_.predict(np.asarray(X, dtype=float)), dtype=float).ravel()


class IdentityTransformer:
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IdentityTransformer":
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if input_features is None:
            input_features = [f"x{i+1}" for i in range(self.n_features_in_)]
        return np.asarray(list(input_features), dtype=object)


class BaseLearnerAdapter:
    def __init__(
        self,
        learner_key: str,
        params: Dict[str, Any],
        random_state: int = RANDOM_STATE,
        task_type: str = "regression",
    ):
        self.learner_key = learner_key
        self.params = dict(params)
        self.random_state = random_state
        self.task_type = task_type

    def _make_transformer(self):
        if self.learner_key in (
            "ridge_linear",
            "elasticnet_linear",
            "logistic_regression",
            "decision_tree",
            "tree_shallow",
            "random_forest",
            "xgboost",
            "lightgbm",
            "catboost",
            "gam",
            "ebm",
            "ga2m",
            "mlp",
        ):
            return IdentityTransformer()
        if self.learner_key in ("ridge_quadratic", "elasticnet_quadratic"):
            return PolynomialFeatures(degree=2, include_bias=False)
        if self.learner_key == "spline_ridge":
            n_knots = int(self.params.get("n_knots", 5))
            return SplineTransformer(
                n_knots=n_knots,
                degree=3,
                include_bias=False,
                extrapolation="linear",
            )
        raise ValueError(f"Unknown learner_key={self.learner_key!r}")

    def _make_estimator(self):
        if self.task_type == "classification":
            if self.learner_key == "logistic_regression":
                return LogisticRegression(
                    C=float(self.params.get("C", 1.0)),
                    max_iter=int(self.params.get("max_iter", 5000)),
                    solver=str(self.params.get("solver", "lbfgs")),
                    random_state=self.random_state,
                )
            if self.learner_key in ("decision_tree", "tree_shallow"):
                return DecisionTreeClassifier(
                    max_depth=self.params.get("max_depth", 3),
                    min_samples_leaf=int(self.params.get("min_samples_leaf", 4)),
                    random_state=self.random_state,
                )
            if self.learner_key == "random_forest":
                return RandomForestClassifier(
                    n_estimators=int(self.params.get("n_estimators", 200)),
                    max_depth=self.params.get("max_depth", None),
                    min_samples_leaf=int(self.params.get("min_samples_leaf", 1)),
                    random_state=self.random_state,
                    n_jobs=1,
                )
            if self.learner_key == "xgboost" and HAS_XGB:
                return xgb.XGBClassifier(
                    n_estimators=int(self.params.get("n_estimators", 200)),
                    learning_rate=float(self.params.get("learning_rate", 0.05)),
                    max_depth=int(self.params.get("max_depth", 4)),
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=1,
                )
            if self.learner_key == "lightgbm" and HAS_LGBM:
                return lgb.LGBMClassifier(
                    n_estimators=int(self.params.get("n_estimators", 200)),
                    learning_rate=float(self.params.get("learning_rate", 0.05)),
                    num_leaves=int(self.params.get("num_leaves", 31)),
                    random_state=self.random_state,
                    verbose=-1,
                )
            if self.learner_key == "catboost" and HAS_CATBOOST:
                return CatBoostClassifier(
                    iterations=int(self.params.get("iterations", 200)),
                    learning_rate=float(self.params.get("learning_rate", 0.05)),
                    depth=int(self.params.get("depth", 6)),
                    loss_function="Logloss",
                    verbose=False,
                    random_seed=self.random_state,
                    allow_writing_files=False,
                )
            if self.learner_key == "mlp":
                return MLPClassifier(
                    hidden_layer_sizes=self.params.get("hidden_layer_sizes", (64,)),
                    alpha=float(self.params.get("alpha", 0.0001)),
                    learning_rate_init=float(self.params.get("learning_rate_init", 0.001)),
                    max_iter=int(self.params.get("max_iter", 500)),
                    early_stopping=True,
                    random_state=self.random_state,
                )
            raise ValueError(f"Unknown classification learner_key={self.learner_key!r}")

        if self.learner_key.startswith("ridge_") or self.learner_key == "spline_ridge":
            return Ridge(alpha=float(self.params.get("alpha", 1.0)))
        if self.learner_key.startswith("elasticnet_"):
            return ElasticNet(
                alpha=float(self.params.get("alpha", 0.01)),
                l1_ratio=float(self.params.get("l1_ratio", 0.5)),
                max_iter=10000,
                random_state=self.random_state,
            )
        if self.learner_key in ("decision_tree", "tree_shallow"):
            return DecisionTreeRegressor(
                max_depth=self.params.get("max_depth", 3),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 4)),
                random_state=self.random_state,
            )
        if self.learner_key == "random_forest":
            return RandomForestRegressor(
                n_estimators=int(self.params.get("n_estimators", 200)),
                max_depth=self.params.get("max_depth", None),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 1)),
                random_state=self.random_state,
                n_jobs=1,
            )
        if self.learner_key == "gam":
            return GAMAdditiveRidgeRegressor(
                n_knots=int(self.params.get("n_knots", 5)),
                alpha=float(self.params.get("alpha", 1.0)),
                degree=3,
                random_state=self.random_state,
            )
        if self.learner_key in ("ebm", "ga2m") and HAS_INTERPRET_EBM:
            default_interactions = 10 if self.learner_key == "ga2m" else 0
            return ExplainableBoostingRegressor(
                random_state=self.random_state,
                max_bins=int(self.params.get("max_bins", 256)),
                interactions=int(self.params.get("interactions", default_interactions)),
                smoothing_rounds=int(self.params.get("smoothing_rounds", 256)),
                outer_bags=int(self.params.get("outer_bags", 2)),
                n_jobs=1,
            )
        if self.learner_key == "xgboost" and HAS_XGB:
            return xgb.XGBRegressor(
                n_estimators=int(self.params.get("n_estimators", 200)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                max_depth=int(self.params.get("max_depth", 4)),
                objective="reg:squarederror",
                random_state=self.random_state,
                n_jobs=1,
            )
        if self.learner_key == "lightgbm" and HAS_LGBM:
            return lgb.LGBMRegressor(
                n_estimators=int(self.params.get("n_estimators", 200)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                num_leaves=int(self.params.get("num_leaves", 31)),
                random_state=self.random_state,
                verbose=-1,
            )
        if self.learner_key == "catboost" and HAS_CATBOOST:
            return CatBoostRegressor(
                iterations=int(self.params.get("iterations", 200)),
                learning_rate=float(self.params.get("learning_rate", 0.05)),
                depth=int(self.params.get("depth", 6)),
                loss_function="RMSE",
                verbose=False,
                random_seed=self.random_state,
                allow_writing_files=False,
            )
        if self.learner_key == "mlp":
            return MLPRegressor(
                hidden_layer_sizes=self.params.get("hidden_layer_sizes", (64,)),
                alpha=float(self.params.get("alpha", 0.0001)),
                learning_rate_init=float(self.params.get("learning_rate_init", 0.001)),
                max_iter=int(self.params.get("max_iter", 500)),
                early_stopping=True,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown learner_key={self.learner_key!r}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseLearnerAdapter":
        self.transformer_ = self._make_transformer()
        Xt = self.transformer_.fit_transform(X)
        self.estimator_ = self._make_estimator()
        self.fit_used_sample_weight_ = False
        try:
            self.estimator_.fit(Xt, y, sample_weight=sample_weight)
            self.fit_used_sample_weight_ = sample_weight is not None
        except TypeError:
            self.estimator_.fit(Xt, y)
            if sample_weight is not None:
                _fpel_eprint(
                    f"base learner {self.learner_key!r}: weighted fit raised TypeError; "
                    f"retrying expert fit without sample_weight."
                )
            self.fit_used_sample_weight_ = False
        except Exception as exc:
            msg = str(exc).lower()
            if (
                sample_weight is not None
                and self.learner_key in ("ebm", "ga2m")
                and ("fillweight" in msg or "fill_weight" in msg)
            ):
                _fpel_eprint(
                    f"base learner {self.learner_key!r}: weighted fit failed ({exc!r}); "
                    f"retrying expert fit without sample_weight."
                )
                self.estimator_.fit(Xt, y)
                self.fit_used_sample_weight_ = False
            else:
                raise
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transformer_.transform(X)
        return np.asarray(self.estimator_.predict(Xt), dtype=float).ravel()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self.transformer_.transform(X)
        if hasattr(self.estimator_, "predict_proba"):
            return np.asarray(self.estimator_.predict_proba(Xt), dtype=float)
        pred = self.predict(X)
        classes = np.asarray(getattr(self, "classes_", np.unique(pred)))
        out = np.zeros((len(pred), len(classes)), dtype=float)
        for i, cls in enumerate(classes):
            out[:, i] = pred == cls
        return out

    def feature_summary(self, input_features: Optional[Sequence[str]] = None) -> Dict[str, float]:
        if input_features is None:
            input_features = [f"x{i+1}" for i in range(getattr(self.transformer_, "n_features_in_", 0))]
        if hasattr(self.transformer_, "get_feature_names_out"):
            names = list(self.transformer_.get_feature_names_out(input_features))
        else:
            names = list(input_features)
        est = self.estimator_
        if hasattr(est, "named_steps") and "ridge" in est.named_steps:
            ridge = est.named_steps["ridge"]
            ct = est.named_steps.get("features")
            if hasattr(ridge, "coef_") and ct is not None and hasattr(ct, "get_feature_names_out"):
                try:
                    gam_names = list(ct.get_feature_names_out(input_features))
                except Exception:
                    gam_names = []
                coef = np.asarray(ridge.coef_, dtype=float).ravel()
                if len(gam_names) != coef.size:
                    gam_names = [f"gam_{i+1}" for i in range(coef.size)]
                out = dict(zip(gam_names, coef.tolist()))
                intercept = np.asarray(getattr(ridge, "intercept_", 0.0), dtype=float).ravel()
                out["__intercept__"] = float(intercept.mean()) if intercept.size else 0.0
                return out
        if hasattr(self.estimator_, "coef_"):
            coef = np.asarray(self.estimator_.coef_, dtype=float)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            out = dict(zip(names, coef.ravel().tolist()))
            intercept = np.asarray(getattr(self.estimator_, "intercept_", [0.0]), dtype=float).ravel()
            out["__intercept__"] = float(intercept.mean()) if intercept.size else 0.0
            return out
        if hasattr(self.estimator_, "feature_importances_"):
            return dict(
                zip(names, np.asarray(self.estimator_.feature_importances_, dtype=float).tolist())
            )
        return {}


LocalTabularRegressor = BaseLearnerAdapter


class PureBaselineModel:
    def __init__(
        self,
        learner_key: str,
        params: Dict[str, Any],
        random_state: int = RANDOM_STATE,
        task_type: str = "regression",
    ):
        self.learner_key = learner_key
        self.params = dict(params)
        self.random_state = random_state
        self.task_type = task_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PureBaselineModel":
        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)
        self.model_ = BaseLearnerAdapter(
            self.learner_key,
            self.params,
            self.random_state,
            task_type=self.task_type,
        )
        self.model_.fit(Z, y, sample_weight=None)
        if hasattr(self.model_, "classes_"):
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(X)
        return self.model_.predict(Z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler_.transform(X)
        return self.model_.predict_proba(Z)


PureBaselineRegressor = PureBaselineModel


class ProFPELModel:
    """
    FPEL: feature-profile **soft** partition enhanced learning (``ProFPELModel`` name kept for code continuity).

    Multiple soft feature-profile regions are learned jointly; every local expert
    is trained on all samples with region-specific ``sample_weight`` from **fixed**
    profile-distance gates. Initialization fixes profile **P**, automatic **K** when ``n_regions`` is
    ``None``, ``profile_kmeans`` (or mean) centers, and bandwidth; **W** is ``softmax`` of scaled squared
    distances to those centers. Inner CV runs **once** per fit to build stacked OOF for ``inner_cv_r2``.

    When ``n_regions`` is ``None``, the default ``k_selection="adaptive_gmm_bic"`` grows **K**
    in feature-profile space while Gaussian-mixture **BIC** improves, then stops after a short
    stall; the returned **K** is the best BIC over the explored path (including **K=1**), capped by
    ``min(n - 1, floor(n / p))`` for input feature count ``p``. An explicit integer ``n_regions`` pins **K**
    for fixed-region runs (for example structural ``global_only`` with ``K=1``).

    ``k_selection="gmm_bic"`` uses a bounded grid search over **K** in feature-profile space.
    """

    def __init__(
        self,
        *,
        base_learner: str = PRIMARY_BASE_LEARNER,
        base_params: Optional[Dict[str, Any]] = None,
        task_type: str = "regression",
        n_regions: Optional[int] = None,
        k_selection: str = "adaptive_gmm_bic",
        k_min: int = 1,
        k_max: Any = None,
        min_region_samples: Any = None,
        bandwidth: Any = "auto",
        feature_order: str = "target_corr",
        profile_metric: str = "value_slope",
        slope_weight: float = 1.0,
        inner_cv: int = INNER_CV_FOLDS,
        partition_mode: str = "soft",
        init_strategy: str = "profile_kmeans",
        profile_row_shuffle: bool = False,
        random_state: int = RANDOM_STATE,
        **kwargs: Any,
    ):
        if task_type != "regression":
            raise NotImplementedError("The current FPEL implementation supports regression experiments.")
        self.base_learner = base_learner
        self.base_params = dict(base_params or {})
        self.task_type = task_type
        # n_regions: None -> adaptive GMM-BIC or bounded GMM-BIC per k_selection; int -> fixed K.
        self.n_regions = None if n_regions is None else int(n_regions)
        self.k_selection = str(k_selection)
        self.k_min = int(k_min)
        self.k_max = k_max
        self.min_region_samples = min_region_samples
        self.bandwidth = bandwidth
        self.feature_order = feature_order
        self.profile_metric = profile_metric
        self.slope_weight = float(slope_weight)
        self.inner_cv = int(inner_cv)
        self.partition_mode = partition_mode
        self.init_strategy = init_strategy
        self.profile_row_shuffle = bool(profile_row_shuffle)
        self.random_state = int(random_state)
        # Populated by fit() when K is automatically resolved.
        self._auto_k_diagnostics_: Optional[Dict[str, float]] = None

    def _resolve_n_regions(
        self,
        n_samples: int,
        n_features: int,
        P: Optional[np.ndarray] = None,
    ) -> int:
        """Resolve the number of soft feature-profile regions K."""
        n = int(n_samples)

        if self.n_regions is not None:
            return int(np.clip(int(self.n_regions), 1, max(1, n)))

        if P is None or P.size == 0:
            self._auto_k_diagnostics_ = {
                "method": "fallback_single_region",
                "k_resolved": 1,
                "n_samples": int(n),
            }
            return 1

        P_arr = np.asarray(P, dtype=float)
        if self.k_selection == "adaptive_gmm_bic":
            return self._select_k_adaptive_gmm_bic(P_arr, n_features=int(n_features))
        if self.k_selection == "gmm_bic":
            return self._select_k_gmm_bic(P_arr, n_features=int(n_features))

        self._auto_k_diagnostics_ = {
            "method": "fallback_single_region",
            "k_resolved": 1,
            "n_samples": int(n),
            "k_selection": str(self.k_selection),
        }
        return 1

    def _resolve_min_region_samples(self, n_features: int) -> int:
        if self.min_region_samples is None:
            return 1
        if isinstance(self.min_region_samples, str):
            key = self.min_region_samples.strip().lower()
            if key in ("p", "n_features", "features"):
                return max(1, int(n_features))
        return max(1, int(self.min_region_samples))

    # Each profile region is a local response regime in the p-dimensional input-feature space (a
    # distinct gate over samples in feature-profile coordinates). We cap the maximum number of
    # regions so that, when K is at its upper bound, the average per-region training count
    # n_train / K is on the order of at least p (Omega(p) sample support per region in order of
    # magnitude). Concretely, automatic K search uses GMM-BIC over
    #     1 <= K <= min(n_train - 1, floor(n_train / p)),
    # where n_train is the current training-set size and p is the input feature count; the
    # n_train - 1 term is an additional numeric guard for mixture fitting. Optional ``k_max``
    # can tighten this cap further.
    def _resolve_k_max(self, n_samples: int, n_features: int) -> int:
        """
        Upper **K** for automatic ``adaptive_gmm_bic`` search and for ``gmm_bic`` grid bounds.

        Default cap is ``min(n - 1, floor(n / p))`` with training size ``n`` and input feature count ``p``.
        If ``k_max`` is a positive integer, the cap is ``min(default_cap, k_max)``. Strings
        ``n_over_p``, ``floor_n_over_p``, or ``n/p`` select the same default cap as ``k_max is None``.
        """
        n = int(n_samples)
        p = max(1, int(n_features))
        numeric = max(1, n - 1)
        n_over_p = max(1, n // p)
        baseline_cap = int(min(numeric, n_over_p))
        if self.k_max is None:
            return baseline_cap
        if isinstance(self.k_max, str):
            key = self.k_max.strip().lower()
            if key in ("n_over_p", "floor_n_over_p", "n/p"):
                return baseline_cap
        return min(baseline_cap, max(1, int(self.k_max)))

    def _select_k_adaptive_gmm_bic(self, P: np.ndarray, *, n_features: int) -> int:
        n = int(P.shape[0])
        k_lo = max(1, int(self.k_min))
        hard_cap = self._resolve_k_max(n, int(n_features))
        if k_lo > hard_cap:
            self._auto_k_diagnostics_ = {
                "method": "adaptive_gmm_bic",
                "k_resolved": int(hard_cap),
                "note": "k_min exceeds numeric cap; clipped",
                "n_samples": int(n),
            }
            return int(hard_cap)

        best_bic = float(np.inf)
        best_k = int(k_lo)
        stall = 0
        bic_rows: List[Dict[str, float]] = []
        for k in range(k_lo, hard_cap + 1):
            try:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    reg_covar=1e-6,
                    n_init=3,
                    random_state=self.random_state,
                )
                gmm.fit(P)
                bic = float(gmm.bic(P))
            except Exception:
                bic = float(np.inf)
            bic_rows.append({"k": float(k), "bic": float(bic)})
            if bic < best_bic - 1e-8:
                best_bic = bic
                best_k = int(k)
                stall = 0
            else:
                stall += 1
                if stall >= ADAPTIVE_GMM_BIC_STALL_MAX:
                    break

        self._auto_k_diagnostics_ = {
            "method": "adaptive_gmm_bic",
            "k_min": float(k_lo),
            "hard_cap": float(hard_cap),
            "n_over_p_cap": float(max(1, n // max(1, int(n_features)))),
            "stall_max": float(ADAPTIVE_GMM_BIC_STALL_MAX),
            "k_resolved": int(best_k),
            "best_bic": float(best_bic),
            "n_samples": int(n),
            "profile_dim": float(P.shape[1]),
            "bic_path": json_text(bic_rows),
        }
        return int(best_k)

    def _select_k_gmm_bic(self, P: np.ndarray, *, n_features: int) -> int:
        n = int(P.shape[0])
        k_min = max(1, int(self.k_min))
        min_support = self._resolve_min_region_samples(n_features)
        safe_upper = max(k_min, n // max(1, int(min_support)))
        safe_upper = min(safe_upper, self._resolve_k_max(n, n_features))
        safe_upper = min(safe_upper, max(1, n - 1))
        if safe_upper < k_min:
            safe_upper = k_min

        k_values = list(range(k_min, safe_upper + 1))
        if not k_values:
            self._auto_k_diagnostics_ = {
                "method": "gmm_bic",
                "k_resolved": 1,
                "note": "empty K grid; fallback K=1",
                "n_samples": int(n),
            }
            return 1
        if 1 not in k_values and 1 <= safe_upper:
            k_values = [1] + k_values

        best_k = k_values[0]
        best_bic = np.inf
        bic_rows: List[Dict[str, float]] = []
        for k in k_values:
            try:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type="diag",
                    reg_covar=1e-6,
                    n_init=3,
                    random_state=self.random_state,
                )
                gmm.fit(P)
                bic = float(gmm.bic(P))
            except Exception:
                bic = np.inf
            bic_rows.append({"k": float(k), "bic": float(bic)})
            if bic < best_bic:
                best_bic = bic
                best_k = int(k)

        self._auto_k_diagnostics_ = {
            "method": "gmm_bic",
            "k_min": float(k_min),
            "k_max": float(safe_upper),
            "min_region_samples": float(min_support),
            "k_resolved": int(best_k),
            "best_bic": float(best_bic),
            "n_samples": int(n),
            "bic_path": json_text(bic_rows),
        }
        return int(best_k)

    @classmethod
    def preview_rate_based_K(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        base_learner: str = PRIMARY_BASE_LEARNER,
        profile_metric: str = "value_slope",
        slope_weight: float = 1.0,
        feature_order: str = "target_corr",
    ) -> Dict[str, Any]:
        """Resolve automatic **K** for ``(X, y)`` without calling ``fit`` on a full FPEL model.

        Uses the configured ``k_selection`` (default ``adaptive_gmm_bic``) on the feature-profile
        matrix **P**, matching the production auto-**K** path.
        """
        mdl = cls(
            base_learner=base_learner,
            n_regions=None,
            profile_metric=profile_metric,
            slope_weight=slope_weight,
            feature_order=feature_order,
        )
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n, d = X_arr.shape
        scaler = StandardScaler()
        Z = scaler.fit_transform(X_arr)
        mdl.feature_order_ = mdl._compute_feature_order(Z, y_arr)
        P = mdl._build_profile(Z)
        K = mdl._resolve_n_regions(n, d, P=P)
        return {
            "K": int(K),
            "auto_k_diagnostics": (
                dict(mdl._auto_k_diagnostics_)
                if mdl._auto_k_diagnostics_ is not None
                else None
            ),
        }

    def _compute_feature_order(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.feature_order == "original":
            return np.arange(Z.shape[1])
        if self.feature_order == "target_corr":
            corrs = np.zeros(Z.shape[1], dtype=float)
            for j in range(Z.shape[1]):
                col = Z[:, j]
                if np.std(col) > 1e-12:
                    corrs[j] = abs(np.corrcoef(col, y)[0, 1])
            return np.argsort(-corrs)
        raise ValueError(f"Unknown feature_order={self.feature_order!r}")

    def _build_profile(self, Z: np.ndarray) -> np.ndarray:
        Z_ord = Z[:, self.feature_order_]
        if self.profile_metric == "value":
            return Z_ord
        if self.profile_metric == "value_slope":
            if Z_ord.shape[1] < 2:
                return Z_ord
            slope = np.diff(Z_ord, axis=1) * self.slope_weight
            return np.hstack([Z_ord, slope])
        if self.profile_metric == "slope":
            if Z_ord.shape[1] < 2:
                return np.zeros((Z_ord.shape[0], 1), dtype=float)
            return np.diff(Z_ord, axis=1) * self.slope_weight
        raise ValueError(f"Unknown profile_metric={self.profile_metric!r}")

    def _estimate_bandwidth(self, P: np.ndarray, centers: np.ndarray) -> float:
        d2 = ((P[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        bw = float(np.median(np.sqrt(d2.min(axis=1))))
        return max(bw, 1e-6)

    def _soft_weights(self, P: np.ndarray, centers: np.ndarray, bw: float) -> np.ndarray:
        d2 = ((P[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        W = softmax(-d2 / (2.0 * bw**2))
        if self.partition_mode == "hard":
            hard = np.zeros_like(W)
            hard[np.arange(W.shape[0]), np.argmax(W, axis=1)] = 1.0
            return hard
        if self.partition_mode != "soft":
            raise ValueError(f"Unknown partition_mode={self.partition_mode!r}")
        return W

    @staticmethod
    def _sanitize_expert_sample_weights(wk: np.ndarray) -> np.ndarray:
        """
        Region gate weights can be extremely small when K is large; some learners (Interpret EBM)
        reject non-finite or denormal weights in native ``FillWeight``. Clamp to a small positive floor.
        """
        w = np.asarray(wk, dtype=np.float64, order="C").ravel()
        w = np.nan_to_num(w, copy=False, nan=1e-6, posinf=1.0, neginf=1e-6)
        floor = 1e-6
        w = np.where(w < floor, floor, w)
        return w

    def _fit_local_models(self, Z: np.ndarray, y: np.ndarray, W: np.ndarray) -> List[BaseLearnerAdapter]:
        models: List[BaseLearnerAdapter] = []
        for k in range(W.shape[1]):
            wk = self._sanitize_expert_sample_weights(W[:, k])
            if float(wk.sum()) <= 1e-12:
                wk = np.ones_like(wk, dtype=np.float64)
            model = BaseLearnerAdapter(
                learner_key=self.base_learner,
                params=self.base_params,
                random_state=self.random_state + k,
                task_type=self.task_type,
            )
            model.fit(Z, y, sample_weight=wk)
            models.append(model)
        return models

    @staticmethod
    def _ensemble(local_preds: np.ndarray, W: np.ndarray) -> np.ndarray:
        return np.sum(local_preds * W, axis=1)

    def _predict_locals(self, Z: np.ndarray, models: Sequence[BaseLearnerAdapter]) -> np.ndarray:
        preds = np.zeros((Z.shape[0], len(models)), dtype=float)
        for k, model in enumerate(models):
            preds[:, k] = model.predict(Z)
        return preds

    def _inner_cv(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        P: np.ndarray,
        centers: np.ndarray,
        bw: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = Z.shape[0]
        n_splits = min(self.inner_cv, n)
        if n_splits < 2:
            W = self._soft_weights(P, centers, bw)
            models = self._fit_local_models(Z, y, W)
            local_pred = self._predict_locals(Z, models)
            return self._ensemble(local_pred, W), local_pred

        cv_pred = np.zeros(n, dtype=float)
        local_oof = np.zeros((n, centers.shape[0]), dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(Z):
            W_tr = self._soft_weights(P[tr], centers, bw)
            models = self._fit_local_models(Z[tr], y[tr], W_tr)
            W_va = self._soft_weights(P[va], centers, bw)
            local_va = self._predict_locals(Z[va], models)
            cv_pred[va] = self._ensemble(local_va, W_va)
            local_oof[va] = local_va
        return cv_pred, local_oof

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProFPELModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self.local_experts_used_sample_weight_ = False

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X)
        self.input_feature_names_ = [f"x{i+1}" for i in range(d)]
        self.feature_order_ = self._compute_feature_order(Z, y)
        P = self._build_profile(Z)
        if self.profile_row_shuffle:
            rng_perm = np.random.default_rng(int(self.random_state) + 913615)
            P = np.asarray(P, dtype=float)[rng_perm.permutation(n)]
        K = self._resolve_n_regions(n, d, P=P)
        self.n_regions_ = K

        n_splits_eff = min(self.inner_cv, max(1, n))
        if n_splits_eff >= 2:
            min_tr_size = int(n - int(np.ceil(n / float(n_splits_eff))))
        else:
            min_tr_size = n
        if K > 1 and n < max(30, K * 4):
            _fpel_eprint(
                f"n_samples={n}, n_regions={K}: few samples per region; "
                f"partition gates and inner-CV scores may be high-variance."
            )
        if K > 1 and min_tr_size < max(12, K + d):
            _fpel_eprint(
                f"inner_cv={self.inner_cv} implies min fold train size ~{min_tr_size} "
                f"(n={n}); K={K}, n_features={d}. Fold-local static FPEL fits may be unstable."
            )

        if K == 1:
            centers = P.mean(axis=0, keepdims=True)
        elif self.init_strategy == "profile_kmeans":
            km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
            centers = km.fit(P).cluster_centers_.copy()
        elif self.init_strategy == "random_centers":
            rng = np.random.default_rng(self.random_state)
            centers = P[rng.choice(P.shape[0], size=K, replace=P.shape[0] < K)].copy()
        else:
            raise ValueError(f"Unknown init_strategy={self.init_strategy!r}")

        bw = self._estimate_bandwidth(P, centers) if self.bandwidth == "auto" else float(self.bandwidth)

        cv_pred, local_oof = self._inner_cv(Z, y, P, centers, bw)
        self.inner_cv_r2_ = float(r2_score(y, cv_pred))
        self.inner_cv_rmse_ = rmse(y, cv_pred)
        W = self._soft_weights(P, centers, bw)
        region_r2 = np.array(
            [weighted_r2(y, local_oof[:, k], W[:, k]) for k in range(W.shape[1])],
            dtype=float,
        )
        self.centers_ = centers.copy()
        self.bandwidth_ = float(bw)
        self.region_inner_cv_r2_ = region_r2.copy()

        self.local_models_ = self._fit_local_models(Z, y, W)
        self.local_experts_used_sample_weight_ = bool(
            self.local_models_
            and all(getattr(m, "fit_used_sample_weight_", False) for m in self.local_models_)
        )
        if STRICT_SAMPLE_WEIGHT and not self.local_experts_used_sample_weight_:
            raise RuntimeError(
                "PROFPEL_STRICT_SAMPLE_WEIGHT or FPPEL_STRICT_SAMPLE_WEIGHT requires sample_weight on "
                f"every local expert (base learner {self.base_learner!r}, {len(self.local_models_)} experts)."
            )
        self.region_support_ = W.sum(axis=0)
        local_pred = self._predict_locals(Z, self.local_models_)
        ens_pred = self._ensemble(local_pred, W)
        self.local_r2_ = np.array(
            [weighted_r2(y, local_pred[:, k], W[:, k]) for k in range(W.shape[1])],
            dtype=float,
        )
        self.train_r2_ = float(r2_score(y, ens_pred))
        self.train_rmse_ = rmse(y, ens_pred)

        self.n_regions_selected_ = int(K)
        self.n_components_ = int(K)
        self.n_local_models_ = int(K)
        return self

    def _transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        Z = self.scaler_.transform(X)
        P = self._build_profile(Z)
        return Z, P

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z, P = self._transform(X)
        W = self._soft_weights(P, self.centers_, self.bandwidth_)
        local_pred = self._predict_locals(Z, self.local_models_)
        return self._ensemble(local_pred, W)

    def region_weights(self, X: np.ndarray) -> np.ndarray:
        _, P = self._transform(X)
        return self._soft_weights(P, self.centers_, self.bandwidth_)

    def assign_region(self, X: np.ndarray) -> np.ndarray:
        return self.region_weights(X).argmax(axis=1)

    def get_local_coefficients(
        self,
        feature_names: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, float]]:
        if feature_names is None:
            feature_names = self.input_feature_names_
        return [model.feature_summary(feature_names) for model in self.local_models_]

    def region_summaries(self) -> Dict[str, Any]:
        return {
            "n_regions": int(self.n_regions_),
            "n_regions_mode": "fixed" if self.n_regions is not None else self.k_selection,
            "auto_k_diagnostics": dict(self._auto_k_diagnostics_)
            if self._auto_k_diagnostics_ is not None
            else None,
            "n_regions_selected": int(self.n_regions_selected_),
            "n_components": int(self.n_components_),
            "n_local_models": int(self.n_local_models_),
            "local_experts_used_sample_weight": bool(getattr(self, "local_experts_used_sample_weight_", False)),
            "centers": self.centers_,
            "bandwidth": self.bandwidth_,
            "feature_order": self.feature_order_,
            "region_support": self.region_support_,
            "local_r2_train_weighted": self.local_r2_,
            "region_inner_cv_r2": self.region_inner_cv_r2_,
            "inner_cv_r2_best": self.inner_cv_r2_,
            "inner_cv_rmse_best": self.inner_cv_rmse_,
            "profile_row_shuffle": bool(getattr(self, "profile_row_shuffle", False)),
            "train_r2": self.train_r2_,
            "train_rmse": self.train_rmse_,
        }


FeatureProfilePartitionEnhancedLearning = ProFPELModel
ProFPELRegressor = ProFPELModel
FPPELRegressor = ProFPELModel
CVFPLMRegressor = ProFPELModel


def base_feature_names(n_features: int) -> List[str]:
    return [f"x{i+1}" for i in range(int(n_features))]


def _select_whitebox_base_params_inner_cv(
    learner_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_strata: Optional[Sequence[Any]],
    random_state: int,
) -> Dict[str, Any]:
    """Select base-learner hyperparameters with the Pure model on the benchmark inner-CV splits."""
    candidates = cartesian_grid(BASE_LEARNER_SPECS[learner_key]["grid"])
    cv_splits = make_inner_cv_splits(len(y_train), random_state, inner_strata)
    best: Optional[Dict[str, Any]] = None
    for params in candidates:
        oof = np.zeros(len(y_train), dtype=float)
        for fold_id, (tr, va) in enumerate(cv_splits):
            model = PureBaselineRegressor(learner_key, params, random_state + 17 * fold_id)
            model.fit(X_train[tr], y_train[tr])
            oof[va] = model.predict(X_train[va])
        cand_r2 = float(r2_score(y_train, oof))
        cand_rmse = rmse(y_train, oof)
        if best is None or cand_r2 > best["inner_cv_r2"] + 1e-12 or (
            abs(cand_r2 - best["inner_cv_r2"]) <= 1e-12 and cand_rmse < best["inner_cv_rmse"]
        ):
            best = {
                "params": dict(params),
                "inner_cv_r2": cand_r2,
                "inner_cv_rmse": cand_rmse,
            }
    assert best is not None
    return best


def select_pure_baseline(
    learner_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_strata: Optional[Sequence[Any]],
    random_state: int,
) -> Dict[str, Any]:
    best = _select_whitebox_base_params_inner_cv(
        learner_key, X_train, y_train, inner_strata=inner_strata, random_state=random_state
    )
    fitted = PureBaselineRegressor(learner_key, best["params"], random_state)
    fitted.fit(X_train, y_train)
    return {
        "model": fitted,
        "params": best["params"],
        "inner_cv_r2": best["inner_cv_r2"],
        "inner_cv_rmse": best["inner_cv_rmse"],
    }


def select_tuned_black_box_baseline(
    learner_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_strata: Optional[Sequence[Any]],
    random_state: int,
) -> Dict[str, Any]:
    grid = BLACK_BOX_TUNING_GRIDS.get(learner_key)
    if grid is None:
        raise KeyError(f"No BLACK_BOX_TUNING_GRIDS entry for {learner_key!r}")
    candidates = cartesian_grid(grid)
    cv_splits = make_inner_cv_splits(len(y_train), random_state, inner_strata)
    best: Optional[Dict[str, Any]] = None

    for params in candidates:
        oof = np.zeros(len(y_train), dtype=float)
        for fold_id, (tr, va) in enumerate(cv_splits):
            model = PureBaselineRegressor(learner_key, params, random_state + 17 * fold_id)
            model.fit(X_train[tr], y_train[tr])
            oof[va] = model.predict(X_train[va])
        cand_r2 = float(r2_score(y_train, oof))
        cand_rmse = rmse(y_train, oof)
        if best is None or cand_r2 > best["inner_cv_r2"] + 1e-12 or (
            abs(cand_r2 - best["inner_cv_r2"]) <= 1e-12 and cand_rmse < best["inner_cv_rmse"]
        ):
            fitted = PureBaselineRegressor(learner_key, params, random_state)
            fitted.fit(X_train, y_train)
            best = {
                "model": fitted,
                "params": dict(params),
                "inner_cv_r2": cand_r2,
                "inner_cv_rmse": cand_rmse,
            }
    assert best is not None
    return best


def append_black_box_baseline_rows(
    rows: List[Dict[str, Any]],
    *,
    dataset_name: str,
    target: Optional[str],
    split_id: int,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    inner_strata: Optional[Sequence[Any]],
    split_seed: int,
) -> None:
    for bk in active_black_box_baselines():
        tag = target or dataset_name
        print(f"    [{tag} s{split_id}] black-box baseline {bk} (tuned inner CV)...", flush=True)
        t0 = time.time()
        best = select_tuned_black_box_baseline(
            bk, X_tr, y_tr, inner_strata=inner_strata, random_state=split_seed
        )
        model = best["model"]
        train_pred = model.predict(X_tr)
        test_pred = model.predict(X_te)
        train_scores = score_dict(y_tr, train_pred)
        test_scores = score_dict(y_te, test_pred)
        elapsed = time.time() - t0
        print(
            f"    [{tag} s{split_id}] black-box {bk}: inner_cv_R2={float(best['inner_cv_r2']):.4f} "
            f"train_R2={float(train_scores['R2']):.4f} test_R2={float(test_scores['R2']):.4f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        sp = BASE_LEARNER_SPECS[bk]
        rows.append(
            pack_result_row(
                dataset=dataset_name,
                target=target,
                split=split_id,
                structure="BlackBoxBaseline",
                base_learner=bk,
                model_label=BLACK_BOX_BASELINE_LABELS[bk],
                learner_group="ExternalBlackBoxBaseline",
                model_type="BlackBoxBaseline",
                model_family=sp.get("model_family", ""),
                model_family_label=sp.get("model_family_label", ""),
                base_learner_short=sp.get("short", bk),
                profpel_local_weights_accepted=None,
                train_time=elapsed,
                train_scores=train_scores,
                test_scores=test_scores,
                selected_params=best["params"],
                notes="train-only inner-CV tuned black-box baseline (BLACK_BOX_TUNING_GRIDS; Structure=BlackBoxBaseline)",
            )
        )


def select_profpel(
    learner_key: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    overrides: Optional[Dict[str, Any]] = None,
    inner_strata: Optional[Sequence[Any]] = None,
    task_type: str = "regression",
    supervised_fallback_tol: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Two-stage selection: white-box inner CV locks ``params``; **fold-local static FPEL** fits build stacked OOF.

    Return mapping always includes ``model``, ``params``, ``inner_cv_r2``, ``inner_cv_rmse`` (stacked OOF on
    the training set), ``fold_r2``, ``fold_regions``, ``whitebox_selection_inner_cv_r2``,
    ``whitebox_selection_inner_cv_rmse``, ``supervised_fallback_tol``, ``supervised_pure_replacement`` (bool),
    ``n_regions``, ``n_components``, ``local_experts_used_sample_weight``.

    When the effective margin from ``supervised_fallback_tol`` or ``FPPEL_SUPERVISED_FALLBACK_TOL`` is
    not ``None`` and ``inner_cv_r2 <= whitebox_selection_inner_cv_r2 + margin``, ``supervised_pure_replacement``
    is true, ``model`` is ``PureBaselineRegressor`` on the full training matrix, and there is no full-data
    ``ProFPELModel.fit``. Otherwise ``model`` is ``ProFPELModel`` after a full-data refit.
    """
    if task_type != "regression":
        raise NotImplementedError("select_profpel currently supports regression experiments.")
    overrides = dict(overrides or {})
    model_kwargs = dict(DEFAULT_PROFPEL_KWARGS)
    model_kwargs.update(overrides)
    cv_splits = make_inner_cv_splits(len(y_train), random_state, inner_strata)

    wb = _select_whitebox_base_params_inner_cv(
        learner_key, X_train, y_train, inner_strata=inner_strata, random_state=random_state
    )
    params: Dict[str, Any] = dict(wb["params"])
    wb_sel_r2 = float(wb["inner_cv_r2"])
    wb_sel_rmse = float(wb["inner_cv_rmse"])

    oof = np.zeros(len(y_train), dtype=float)
    fold_scores: List[float] = []
    fold_regions: List[int] = []
    for fold_id, (tr, va) in enumerate(cv_splits):
        fold_model = ProFPELModel(
            base_learner=learner_key,
            base_params=params,
            task_type=task_type,
            random_state=random_state + 1009 * (fold_id + 1),
            **model_kwargs,
        )
        fold_model.fit(X_train[tr], y_train[tr])
        oof[va] = fold_model.predict(X_train[va])
        fold_scores.append(float(r2_score(y_train[va], oof[va])))
        fold_regions.append(int(fold_model.n_regions_selected_))
    inner_r2 = float(r2_score(y_train, oof))
    inner_rmse = rmse(y_train, oof)
    fb_tol = _resolve_supervised_fallback_tol(supervised_fallback_tol)
    best = {
        "params": dict(params),
        "inner_cv_r2": inner_r2,
        "inner_cv_rmse": inner_rmse,
        "fold_r2": np.asarray(fold_scores, dtype=float),
        "fold_regions": np.asarray(fold_regions, dtype=int),
        "overrides": dict(overrides),
        "whitebox_selection_inner_cv_r2": wb_sel_r2,
        "whitebox_selection_inner_cv_rmse": wb_sel_rmse,
        "supervised_fallback_tol": fb_tol,
        "supervised_pure_replacement": False,
        "selection_protocol": (
            "FPEL-B repeats the same Pure-equivalent white-box inner CV, fixes the winner, then runs "
            "fold-local static FPEL fits for stacked OOF and a full-training refit"
        ),
    }
    if fb_tol is not None and inner_r2 <= wb_sel_r2 + float(fb_tol):
        final_model = PureBaselineRegressor(learner_key, best["params"], random_state, task_type=task_type)
        final_model.fit(X_train, y_train)
        final_model.n_regions_selected_ = 1
        final_model.n_components_ = 1
        final_model.strict_selection_oof_r2_ = float(inner_r2)
        final_model.strict_selection_oof_rmse_ = float(inner_rmse)
        final_model.strict_selection_fold_r2_ = np.asarray(best["fold_r2"], dtype=float)
        final_model.supervised_pure_replacement_ = True
        final_model.whitebox_selection_inner_cv_r2_ = float(wb_sel_r2)
        final_model.local_experts_used_sample_weight_ = False
        best["supervised_pure_replacement"] = True
        best.update(
            {
                "model": final_model,
                "n_regions": 1,
                "n_components": 1,
                "local_experts_used_sample_weight": False,
            }
        )
        return best

    final_model = ProFPELModel(
        base_learner=learner_key,
        base_params=best["params"],
        task_type=task_type,
        random_state=random_state,
        **model_kwargs,
    )
    final_model.fit(X_train, y_train)
    final_model.strict_selection_oof_r2_ = best["inner_cv_r2"]
    final_model.strict_selection_oof_rmse_ = best["inner_cv_rmse"]
    final_model.strict_selection_fold_r2_ = np.asarray(best["fold_r2"], dtype=float)
    final_model.supervised_pure_replacement_ = False
    final_model.whitebox_selection_inner_cv_r2_ = float(wb_sel_r2)
    best.update(
        {
            "model": final_model,
            "n_regions": int(final_model.n_regions_selected_),
            "n_components": int(final_model.n_components_),
            "local_experts_used_sample_weight": bool(final_model.local_experts_used_sample_weight_),
        }
    )
    return best


def locate_biomass_workbook() -> str:
    for path in BIOMASS_WORKBOOK_CANDIDATES:
        if path and os.path.exists(path):
            return path
    matches = list(Path(WORKSPACE_DIR).rglob(BIOMASS_WORKBOOK_NAME))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(
        f"Could not locate {BIOMASS_WORKBOOK_NAME!r}. "
        "Set FPPEL_BIOMASS_PATH to the workbook path if it is outside the workspace."
    )


def locate_workbook(name: str, candidates: Sequence[str], env_var: str) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    matches = list(Path(WORKSPACE_DIR).rglob(name))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(
        f"Could not locate {name!r}. Set {env_var} to the workbook path if it is outside the workspace."
    )


def load_hhv_data() -> pd.DataFrame:
    path = locate_workbook(HHV_WORKBOOK_NAME, HHV_WORKBOOK_CANDIDATES, "FPPEL_HHV_PATH")
    df = pd.read_excel(path, sheet_name="Sheet1", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    keep = HHV_FEATURE_COLUMNS + HHV_TARGETS
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise KeyError(f"Missing HHV columns: {missing}")
    out = df[keep].copy()
    for col in keep:
        out[col] = coerce_numeric_series(out[col])
    return out


def load_co_gasification_data() -> pd.DataFrame:
    path = locate_workbook(CO_GAS_WORKBOOK_NAME, CO_GAS_WORKBOOK_CANDIDATES, "FPPEL_CO_GAS_PATH")
    df = pd.read_excel(path, sheet_name="Datasheet", header=3)
    if df.shape[1] < len(CO_GAS_COLUMNS):
        raise KeyError(
            f"Co-gasification sheet has {df.shape[1]} columns, expected at least {len(CO_GAS_COLUMNS)}."
        )
    df = df.iloc[:, : len(CO_GAS_COLUMNS)].copy()
    df.columns = CO_GAS_COLUMNS
    numeric_cols = CO_GAS_FEATURE_COLUMNS + CO_GAS_TARGETS
    for col in numeric_cols:
        df[col] = coerce_numeric_series(df[col])
    return df


def load_biomass_data() -> pd.DataFrame:
    path = locate_biomass_workbook()
    df_pct = pd.read_excel(path, sheet_name=SHEET_PERCENT)
    df_gy = pd.read_excel(path, sheet_name=SHEET_GY)
    df_pct["__id__"] = range(len(df_pct))
    df_gy["__id__"] = range(len(df_gy))
    df = pd.merge(df_pct, df_gy, on="__id__", how="outer", suffixes=("", "_GY")).drop(columns=["__id__"])
    df.columns = [str(c).strip() for c in df.columns]

    def _find(cands: Sequence[str]) -> Optional[str]:
        for cand in cands:
            if cand in df.columns:
                return cand
        return None

    col_map: Dict[str, str] = {}
    for internal, cands in _COL_CANDIDATES.items():
        found = _find(cands)
        if found is None and internal == "T":
            for col in df.columns:
                if str(col).startswith("(x1)T"):
                    found = col
                    break
        if found is not None and found != internal:
            col_map[found] = internal

    for short, real in TARGETS_MAP.items():
        if real in df.columns and real != short:
            col_map[real] = short

    bed_found = _find((BED_COL, "bed material"))
    if bed_found is None:
        for col in df.columns:
            if "bed" in str(col).lower():
                bed_found = col
                break
    if bed_found is None:
        raise KeyError("Could not locate bed-material column for biomass dataset.")
    if bed_found != BED_COL:
        col_map[bed_found] = BED_COL

    df = df.rename(columns=col_map)
    if "T" in df.columns:
        df["T"] = pd.to_numeric(df["T"], errors="coerce") + 273.15
    missing = [col for col in CONTINUOUS_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing biomass columns: {missing}")

    df[BED_LABEL_COL] = df[BED_COL].astype(str)
    dummies = pd.get_dummies(df[BED_COL].astype(str), prefix="Bed", drop_first=False)
    for val in BED_MATERIAL_VALUES:
        col = f"Bed_{val}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[BED_DUMMIES].astype(int)
    df = pd.concat([df.drop(columns=[BED_COL]), dummies], axis=1)
    return df


def prepare_biomass_target(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    part = df.dropna(subset=CONTINUOUS_COLS + [target]).copy().reset_index(drop=True)
    X = part[FEATURE_COLUMNS].astype(float).values
    y = part[target].astype(float).values
    strata = part[BED_LABEL_COL].astype(str).values
    return X, y, strata, FEATURE_COLUMNS.copy(), part


def prepare_tabular_target(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    needed = list(feature_columns) + [target]
    part = df.dropna(subset=needed).copy().reset_index(drop=True)
    X = part[list(feature_columns)].astype(float).values
    y = part[target].astype(float).values
    return X, y, list(feature_columns), part


def build_outer_splits(
    n_samples: int,
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = ShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    idx = np.arange(int(n_samples))
    return list(splitter.split(idx))


def build_outer_splits_from_strata(
    strata: Sequence[Any],
    *,
    n_splits: int,
    test_size: float,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    strata_arr = np.asarray(strata)
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )
    idx = np.arange(len(strata_arr))
    try:
        return list(splitter.split(idx, strata_arr))
    except ValueError:
        kf = KFold(
            n_splits=min(int(n_splits), len(idx)),
            shuffle=True,
            random_state=random_state,
        )
        return list(kf.split(idx))


def pack_result_row(
    *,
    dataset: str,
    target: Optional[str],
    split: int,
    structure: str,
    base_learner: Optional[str],
    model_label: str,
    learner_group: str,
    train_time: float,
    train_scores: Dict[str, float],
    test_scores: Dict[str, float],
    selected_params: Optional[Dict[str, Any]] = None,
    selected_n_regions: Optional[int] = None,
    structural_delta_eligible: Optional[bool] = None,
    notes: Optional[str] = None,
    model_type: Optional[str] = None,
    model_family: Optional[str] = None,
    model_family_label: Optional[str] = None,
    base_learner_short: Optional[str] = None,
    profpel_local_weights_accepted: Optional[bool] = None,
) -> Dict[str, Any]:
    short = base_learner_short if base_learner_short is not None else model_type
    row = {
        "Dataset": dataset,
        "Target": target,
        "Split": int(split),
        "Structure": structure,
        "BaseLearner": base_learner,
        "Model": model_label,
        "LearnerGroup": learner_group,
        "ModelFamily": model_family or "",
        "ModelFamilyLabel": model_family_label or "",
        "BaseLearnerShort": base_learner_short or "",
        "ModelType": short or "",
        "Train_Time": float(train_time),
        "Train_R2": float(train_scores["R2"]),
        "Train_RMSE": float(train_scores["RMSE"]),
        "Train_MAE": float(train_scores["MAE"]),
        "R2": float(test_scores["R2"]),
        "RMSE": float(test_scores["RMSE"]),
        "MAE": float(test_scores["MAE"]),
        "TrainTest_Gap_R2": float(train_scores["R2"] - test_scores["R2"]),
        "SelectedParams": json_text(selected_params) if selected_params is not None else None,
        "Selected_n_regions": float(selected_n_regions) if selected_n_regions is not None else np.nan,
        "StructuralDeltaEligible": (
            bool(structural_delta_eligible) if structural_delta_eligible is not None else np.nan
        ),
        "ProFPEL_LocalWeightsAccepted": (
            float(profpel_local_weights_accepted)
            if isinstance(profpel_local_weights_accepted, bool)
            else np.nan
        ),
        "Notes": notes,
    }
    return row


def collect_primary_coefficients(
    model: ProFPELModel,
    *,
    dataset: str,
    target: str,
    split: int,
    feature_names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    coeffs = model.get_local_coefficients(feature_names)
    summary = model.region_summaries()
    rs = np.asarray(summary["region_support"], dtype=float).ravel()
    n_reg = int(summary["n_regions_selected"])
    for region_idx, coef_map in enumerate(coeffs):
        sup = float(rs[region_idx]) if region_idx < rs.size else float("nan")
        for term_name, coef in coef_map.items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Target": target,
                    "Split": int(split),
                    "Model": BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER]["cv_label"],
                    "RegionIndex": int(region_idx),
                    "Term": term_name,
                    "Value": float(coef),
                    "RegionSupport": sup,
                    "NRegionsSelected": n_reg,
                }
            )
    return rows


def collect_primary_region_summary(
    model: ProFPELModel,
    *,
    dataset: str,
    target: str,
    split: int,
) -> List[Dict[str, Any]]:
    summary = model.region_summaries()
    rows: List[Dict[str, Any]] = []
    rs = np.asarray(summary["region_support"], dtype=float).ravel()
    n_reg = int(summary["n_regions_selected"])
    for region_idx in range(len(rs)):
        rows.append(
            {
                "Dataset": dataset,
                "Target": target,
                "Split": int(split),
                "RegionIndex": int(region_idx),
                "NRegionsSelected": n_reg,
                "RegionSupport": float(rs[region_idx]),
                "GlobalTrainR2": float(summary["train_r2"]),
            }
        )
    return rows


def run_one_learner_pair(
    *,
    dataset_name: str,
    target: Optional[str],
    split_id: int,
    learner_key: str,
    learner_idx: int,
    n_learners: int,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    inner_strata: Optional[Sequence[Any]],
    split_seed: int,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    logs: List[str] = []
    rows: List[Dict[str, Any]] = []
    coeff_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []
    spec = BASE_LEARNER_SPECS[learner_key]
    n_param = len(cartesian_grid(spec["grid"]))
    prefix = f"[{target or dataset_name} s{split_id}]"
    print(
        f"    {prefix} START learner {learner_idx}/{n_learners} {learner_key} (n_param={n_param})",
        flush=True,
    )
    logs.append(
        f"    {prefix} learner {learner_idx}/{n_learners} "
        f"{learner_key}: pure CV over {n_param} params"
    )

    t0 = time.time()
    pure = select_pure_baseline(
        learner_key,
        X_tr,
        y_tr,
        inner_strata=inner_strata,
        random_state=split_seed,
    )
    pure_train_pred = pure["model"].predict(X_tr)
    pure_test_pred = pure["model"].predict(X_te)
    pure_train_scores = score_dict(y_tr, pure_train_pred)
    pure_test_scores = score_dict(y_te, pure_test_pred)
    pure_elapsed = time.time() - t0
    print(
        f"    {prefix} Pure {learner_key}: inner_cv_R2={float(pure['inner_cv_r2']):.4f} "
        f"train_R2={float(pure_train_scores['R2']):.4f} test_R2={float(pure_test_scores['R2']):.4f} "
        f"time={pure_elapsed:.1f}s",
        flush=True,
    )
    rows.append(
        pack_result_row(
            dataset=dataset_name,
            target=target,
            split=split_id,
            structure="Pure",
            base_learner=learner_key,
            model_label=spec["pure_label"],
            learner_group="MainBaseLearner",
            model_type=spec.get("short"),
            model_family=spec.get("model_family"),
            model_family_label=spec.get("model_family_label"),
            base_learner_short=spec.get("short"),
            profpel_local_weights_accepted=None,
            train_time=pure_elapsed,
            train_scores=pure_train_scores,
            test_scores=pure_test_scores,
            selected_params=pure["params"],
            notes="train-only inner-CV baseline selection",
        )
    )

    logs.append(
        f"    {prefix} learner {learner_idx}/{n_learners} "
        f"{learner_key}: FPEL full pipeline (Pure-equivalent white-box inner CV, then fold-local static FPEL OOF)"
    )
    t0 = time.time()
    profpel = select_profpel(
        learner_key,
        X_tr,
        y_tr,
        random_state=split_seed,
        inner_strata=inner_strata,
    )
    profpel_model = profpel["model"]
    profpel_train_pred = profpel_model.predict(X_tr)
    profpel_test_pred = profpel_model.predict(X_te)
    profpel_train_scores = score_dict(y_tr, profpel_train_pred)
    profpel_test_scores = score_dict(y_te, profpel_test_pred)
    _nr = int(profpel["n_regions"])
    profpel_elapsed = time.time() - t0
    _pfb = bool(profpel.get("supervised_pure_replacement", False))
    _fb_tol = profpel.get("supervised_fallback_tol")
    print(
        f"    {prefix} FPEL {learner_key}: inner_cv_R2={float(profpel['inner_cv_r2']):.4f} "
        f"train_R2={float(profpel_train_scores['R2']):.4f} test_R2={float(profpel_test_scores['R2']):.4f} "
        f"time={profpel_elapsed:.1f}s K={_nr}"
        + (" supervised_pure_replacement" if _pfb else ""),
        flush=True,
    )
    logs.append(
        f"    {prefix} {learner_key}: selected "
        f"K={profpel['n_regions']} params={profpel['params']} "
        f"strict_oof_r2={profpel['inner_cv_r2']:.4f}"
        + (
            f" supervised_pure_replacement tol={_fb_tol} whitebox_sel_r2={profpel['whitebox_selection_inner_cv_r2']:.4f}"
            if _pfb
            else ""
        )
    )
    reason_text = "soft_partition_regions"
    partition_detail_lines: List[str] = []
    logs.append(f"    {prefix} {learner_key}: partition diagnostics {reason_text}")
    if partition_detail_lines:
        logs.append(f"    {prefix} {learner_key}: partition_details=" + " | ".join(partition_detail_lines))
    logs.append(
        f"    {prefix} {learner_key}: "
        f"Pure test R2={pure_test_scores['R2']:.4f}, "
        f"FPEL test R2={profpel_test_scores['R2']:.4f}, "
        f"Delta={profpel_test_scores['R2'] - pure_test_scores['R2']:+.4f}"
    )
    _lw_h = getattr(profpel_model, "local_experts_used_sample_weight_", None)
    _notes_pf = (
        "train-only inner-CV: white-box stage (Pure-equivalent) plus fold-local static FPEL stage; "
        "Train_Time is wall-clock for full FPEL pipeline"
    )
    if _pfb:
        _notes_pf += (
            f"; supervised_pure_replacement: stacked fold-local FPEL OOF R2={float(profpel['inner_cv_r2']):.6g} <= "
            f"whitebox_sel_R2={float(profpel['whitebox_selection_inner_cv_r2']):.6g} + tol={_fb_tol}; "
            "final estimator is Pure-B on locked params"
        )
    rows.append(
        pack_result_row(
            dataset=dataset_name,
            target=target,
            split=split_id,
            structure="ProFPEL",
            base_learner=learner_key,
            model_label=spec["cv_label"],
            learner_group="MainBaseLearner",
            model_type=spec.get("short"),
            model_family=spec.get("model_family"),
            model_family_label=spec.get("model_family_label"),
            base_learner_short=spec.get("short"),
            profpel_local_weights_accepted=(bool(_lw_h) if _lw_h is not None else None),
            train_time=profpel_elapsed,
            train_scores=profpel_train_scores,
            test_scores=profpel_test_scores,
            selected_params=profpel["params"],
            selected_n_regions=_nr,
            structural_delta_eligible=_nr > 1,
            notes=_notes_pf,
        )
    )

    if learner_key == PRIMARY_BASE_LEARNER and feature_names is not None and isinstance(
        profpel_model, ProFPELModel
    ):
        coeff_rows.extend(
            collect_primary_coefficients(
                profpel_model,
                dataset=dataset_name,
                target=str(target),
                split=split_id,
                feature_names=feature_names,
            )
        )
        region_rows.extend(
            collect_primary_region_summary(
                profpel_model,
                dataset=dataset_name,
                target=str(target),
                split=split_id,
            )
        )
    return rows, coeff_rows, region_rows, logs


def _flush_learner_task_logs(logs: Sequence[str]) -> None:
    for line in logs:
        print(line, flush=True)


def _run_one_learner_pair_indexed(
    idx: int, kwargs: Dict[str, Any]
) -> Tuple[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]]:
    """Picklable top-level callable for joblib/loky; yields a stable index for reassembly."""
    return idx, run_one_learner_pair(**kwargs)


def run_learner_tasks_parallel(task_kwargs: List[Dict[str, Any]], n_jobs: int):
    """Run learner tasks in parallel; flush each task's log lines as soon as it completes."""
    n_jobs = max(1, int(n_jobs))
    n_task = len(task_kwargs)
    if n_task == 0:
        return []

    if n_jobs == 1 or n_task == 1:
        out = [run_one_learner_pair(**kwargs) for kwargs in task_kwargs]
        for _rp, _cf, _rg, logs in out:
            _flush_learner_task_logs(logs)
        return out

    results: List[Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]]] = [
        None
    ] * n_task

    if PARALLEL_BACKEND == "joblib":
        try:
            gen = Parallel(n_jobs=n_jobs, backend="loky", verbose=0, return_as="generator_unordered")(
                delayed(_run_one_learner_pair_indexed)(i, kwargs)
                for i, kwargs in enumerate(task_kwargs)
            )
            for idx, chunk in gen:
                _rp, _cf, _rg, logs = chunk
                _flush_learner_task_logs(logs)
                results[idx] = chunk
            if all(r is not None for r in results):
                return results  # type: ignore[return-value]
        except TypeError:
            # joblib without return_as / generator_unordered
            pass

    future_to_idx = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for i, kwargs in enumerate(task_kwargs):
            fut = ex.submit(run_one_learner_pair, **kwargs)
            future_to_idx[fut] = i
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            chunk = fut.result()
            _rp, _cf, _rg, logs = chunk
            _flush_learner_task_logs(logs)
            results[i] = chunk
    return results  # type: ignore[return-value]


def run_biomass_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78, flush=True)
    print("Biomass: white-box Pure-B vs FPEL-B; tuned black-box baselines", flush=True)
    print("=" * 78, flush=True)

    df = load_biomass_data()
    rows: List[Dict[str, Any]] = []
    coeff_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []

    for target in TARGETS:
        X, y, outer_strata, feature_names, _ = prepare_biomass_target(df, target)
        splits = build_outer_splits_from_strata(
            outer_strata,
            n_splits=OUTER_SPLITS,
            test_size=BIOMASS_TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        learners = active_main_base_learners()
        print(
            f"[Biomass] Target={target:<4} n={len(y):>4} p={X.shape[1]:>2} splits={len(splits)} "
            f"whitebox={len(learners)} blackbox={len(active_black_box_baselines())}",
            flush=True,
        )
        for split_id, (tr, te) in enumerate(splits, start=1):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            inner_strata = np.asarray(outer_strata)[tr]
            split_seed = RANDOM_STATE + 100 * TARGETS.index(target) + split_id
            k_auto_primary = ProFPELModel.preview_rate_based_K(
                X_tr, y_tr, base_learner=PRIMARY_BASE_LEARNER
            )["K"]
            print(
                f"  [Biomass] {target} split {split_id}/{len(splits)} "
                f"K_auto(primary)={k_auto_primary} "
                f"profile={DEFAULT_PROFPEL_KWARGS['profile_metric']} "
                f"inner_cv_folds={DEFAULT_PROFPEL_KWARGS['inner_cv']}",
                flush=True,
            )

            n_jobs = max(1, min(PARALLEL_N_JOBS, len(learners)))
            print(f"    [{target} s{split_id}] running learners with n_jobs={n_jobs}", flush=True)
            task_kwargs = [
                dict(
                    dataset_name="BiomassGasification",
                    target=target,
                    split_id=split_id,
                    learner_key=learner_key,
                    learner_idx=learner_idx,
                    n_learners=len(learners),
                    X_tr=X_tr,
                    X_te=X_te,
                    y_tr=y_tr,
                    y_te=y_te,
                    inner_strata=inner_strata,
                    split_seed=split_seed,
                    feature_names=feature_names,
                )
                for learner_idx, learner_key in enumerate(learners, start=1)
            ]
            learner_results = run_learner_tasks_parallel(task_kwargs, n_jobs)
            for rows_part, coeff_part, region_part, _logs in learner_results:
                rows.extend(rows_part)
                coeff_rows.extend(coeff_part)
                region_rows.extend(region_part)

            append_black_box_baseline_rows(
                rows,
                dataset_name="BiomassGasification",
                target=target,
                split_id=split_id,
                X_tr=X_tr,
                X_te=X_te,
                y_tr=y_tr,
                y_te=y_te,
                inner_strata=inner_strata,
                split_seed=split_seed,
            )

    return pd.DataFrame(rows), pd.DataFrame(coeff_rows), pd.DataFrame(region_rows)


def run_tabular_multi_target_experiment(
    *,
    dataset_name: str,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    targets: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78, flush=True)
    print(f"{dataset_name}: white-box Pure-B vs FPEL-B; tuned black-box baselines", flush=True)
    print("=" * 78, flush=True)

    rows: List[Dict[str, Any]] = []
    coeff_rows: List[Dict[str, Any]] = []
    region_rows: List[Dict[str, Any]] = []

    for target_idx, target in enumerate(targets):
        X, y, feature_names, _ = prepare_tabular_target(
            df,
            feature_columns=feature_columns,
            target=target,
        )
        splits = build_outer_splits(
            len(y),
            n_splits=OUTER_SPLITS,
            test_size=BIOMASS_TEST_SIZE,
            random_state=RANDOM_STATE + 1000 + target_idx,
        )
        learners = active_main_base_learners()
        print(
            f"[{dataset_name}] Target={target:<12} n={len(y):>4} p={X.shape[1]:>2} "
            f"splits={len(splits)} whitebox={len(learners)} blackbox={len(active_black_box_baselines())}",
            flush=True,
        )
        for split_id, (tr, te) in enumerate(splits, start=1):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            split_seed = RANDOM_STATE + 10000 + 100 * target_idx + split_id
            k_auto_primary = ProFPELModel.preview_rate_based_K(
                X_tr, y_tr, base_learner=PRIMARY_BASE_LEARNER
            )["K"]
            print(
                f"  [{dataset_name}] {target} split {split_id}/{len(splits)} "
                f"K_auto(primary)={k_auto_primary} "
                f"profile={DEFAULT_PROFPEL_KWARGS['profile_metric']} "
                f"inner_cv_folds={DEFAULT_PROFPEL_KWARGS['inner_cv']}",
                flush=True,
            )

            n_jobs = max(1, min(PARALLEL_N_JOBS, len(learners)))
            print(f"    [{target} s{split_id}] running learners with n_jobs={n_jobs}", flush=True)
            task_kwargs = [
                dict(
                    dataset_name=dataset_name,
                    target=target,
                    split_id=split_id,
                    learner_key=learner_key,
                    learner_idx=learner_idx,
                    n_learners=len(learners),
                    X_tr=X_tr,
                    X_te=X_te,
                    y_tr=y_tr,
                    y_te=y_te,
                    inner_strata=None,
                    split_seed=split_seed,
                    feature_names=feature_names,
                )
                for learner_idx, learner_key in enumerate(learners, start=1)
            ]
            learner_results = run_learner_tasks_parallel(task_kwargs, n_jobs)
            for rows_part, coeff_part, region_part, _logs in learner_results:
                rows.extend(rows_part)
                coeff_rows.extend(coeff_part)
                region_rows.extend(region_part)

            append_black_box_baseline_rows(
                rows,
                dataset_name=dataset_name,
                target=target,
                split_id=split_id,
                X_tr=X_tr,
                X_te=X_te,
                y_tr=y_tr,
                y_te=y_te,
                inner_strata=None,
                split_seed=split_seed,
            )

    return pd.DataFrame(rows), pd.DataFrame(coeff_rows), pd.DataFrame(region_rows)


def run_hhv_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_tabular_multi_target_experiment(
        dataset_name="HHV",
        df=load_hhv_data(),
        feature_columns=HHV_FEATURE_COLUMNS,
        targets=HHV_TARGETS,
    )


def run_co_gasification_experiment() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_tabular_multi_target_experiment(
        dataset_name="CoGasification",
        df=load_co_gasification_data(),
        feature_columns=CO_GAS_FEATURE_COLUMNS,
        targets=CO_GAS_TARGETS,
    )


def run_primary_ablation() -> pd.DataFrame:
    print("\n" + "=" * 78, flush=True)
    print("Primary-model ablation: FPEL-Ridge-Q", flush=True)
    print("=" * 78, flush=True)

    rows: List[Dict[str, Any]] = []
    ablation_variants = [
        ("adaptive_K", {}, "region_structure"),
        ("global_only_K1", {"n_regions": 1}, "region_structure"),
        ("value_slope", {"profile_metric": "value_slope"}, "profile_information"),
        ("value_only", {"profile_metric": "value"}, "profile_information"),
        ("slope_only", {"profile_metric": "slope"}, "profile_information"),
        ("shuffled_profile", {"profile_row_shuffle": True}, "profile_information"),
        ("soft_gate", {"partition_mode": "soft"}, "partition_gate"),
        ("hard_partition", {"partition_mode": "hard"}, "partition_gate"),
        ("target_corr_order", {"feature_order": "target_corr"}, "feature_order"),
        ("original_order", {"feature_order": "original"}, "feature_order"),
    ]

    def _canonical_ablation_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
        canon = dict(overrides)
        for key in ("profile_metric", "partition_mode", "feature_order"):
            if canon.get(key) == DEFAULT_PROFPEL_KWARGS.get(key):
                canon.pop(key, None)
        return canon

    def _append_ablation_row(
        *,
        dataset_name: str,
        target: str,
        split_id: int,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        y_tr: np.ndarray,
        y_te: np.ndarray,
        inner_strata: Optional[Sequence[Any]],
        split_seed: int,
        variant_name: str,
        family: str,
        overrides: Dict[str, Any],
        fit_cache: Dict[str, Dict[str, Any]],
    ) -> None:
        canon_overrides = _canonical_ablation_overrides(overrides)
        cache_key = json_text(canon_overrides)
        cached = fit_cache.get(cache_key)
        if cached is None:
            t0 = time.time()
            fit_info = select_profpel(
                PRIMARY_BASE_LEARNER,
                X_tr,
                y_tr,
                random_state=split_seed,
                overrides=canon_overrides,
                inner_strata=inner_strata,
            )
            model = fit_info["model"]
            train_pred = model.predict(X_tr)
            test_pred = model.predict(X_te)
            train_sc = score_dict(y_tr, train_pred)
            test_sc = score_dict(y_te, test_pred)
            abl_elapsed = time.time() - t0
            cached = {
                "fit_info": fit_info,
                "model": model,
                "train_sc": train_sc,
                "test_sc": test_sc,
                "elapsed": abl_elapsed,
            }
            fit_cache[cache_key] = cached
        fit_info = cached["fit_info"]
        model = cached["model"]
        train_sc = cached["train_sc"]
        test_sc = cached["test_sc"]
        abl_elapsed = float(cached["elapsed"])
        _abl_fb = bool(fit_info.get("supervised_pure_replacement", False))
        print(
            f"    [Ablation] {dataset_name} {target} s{split_id} {family}={variant_name}: "
            f"inner_cv_R2={float(fit_info['inner_cv_r2']):.4f} "
            f"train_R2={float(train_sc['R2']):.4f} test_R2={float(test_sc['R2']):.4f} "
            f"time={abl_elapsed:.1f}s K={int(fit_info['n_regions'])}"
            + (" supervised_pure_replacement" if _abl_fb else ""),
            flush=True,
        )
        _abl_lw = getattr(model, "local_experts_used_sample_weight_", None)
        rows.append(
            {
                **pack_result_row(
                    dataset=dataset_name,
                    target=target,
                    split=split_id,
                    structure="Ablation",
                    base_learner=PRIMARY_BASE_LEARNER,
                    model_label=f"Ablation-{variant_name}",
                    learner_group="PrimaryAblation",
                    model_type=BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER].get("short"),
                    model_family=BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER].get("model_family"),
                    model_family_label=BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER].get("model_family_label"),
                    base_learner_short=BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER].get("short"),
                    profpel_local_weights_accepted=(bool(_abl_lw) if _abl_lw is not None else None),
                    train_time=abl_elapsed,
                    train_scores=train_sc,
                    test_scores=test_sc,
                    selected_params=fit_info["params"],
                    selected_n_regions=int(fit_info["n_regions"]),
                    structural_delta_eligible=int(fit_info["n_regions"]) > 1,
                    notes=json_text(canon_overrides),
                ),
                "AblationFamily": family,
                "AblationVariant": variant_name,
            }
        )

    ablation_tasks: List[Dict[str, Any]] = []
    biomass_df = load_biomass_data()
    for target_idx, target in enumerate(TARGETS):
        X, y, outer_strata, _, _ = prepare_biomass_target(biomass_df, target)
        ablation_tasks.append(
            {
                "dataset_name": "BiomassGasification",
                "target": target,
                "target_idx": target_idx,
                "X": X,
                "y": y,
                "outer_strata": outer_strata,
                "splits": build_outer_splits_from_strata(
                    outer_strata,
                    n_splits=OUTER_SPLITS,
                    test_size=BIOMASS_TEST_SIZE,
                    random_state=RANDOM_STATE,
                ),
            }
        )

    hhv_df = load_hhv_data()
    for target_idx, target in enumerate(HHV_TARGETS, start=len(ablation_tasks)):
        X, y, _, _ = prepare_tabular_target(
            hhv_df,
            feature_columns=HHV_FEATURE_COLUMNS,
            target=target,
        )
        ablation_tasks.append(
            {
                "dataset_name": "HHV",
                "target": target,
                "target_idx": target_idx,
                "X": X,
                "y": y,
                "outer_strata": None,
                "splits": build_outer_splits(
                    len(y),
                    n_splits=OUTER_SPLITS,
                    test_size=BIOMASS_TEST_SIZE,
                    random_state=RANDOM_STATE + 1000 + target_idx,
                ),
            }
        )

    co_gas_df = load_co_gasification_data()
    for target_idx, target in enumerate(CO_GAS_TARGETS, start=len(ablation_tasks)):
        X, y, _, _ = prepare_tabular_target(
            co_gas_df,
            feature_columns=CO_GAS_FEATURE_COLUMNS,
            target=target,
        )
        ablation_tasks.append(
            {
                "dataset_name": "CoGasification",
                "target": target,
                "target_idx": target_idx,
                "X": X,
                "y": y,
                "outer_strata": None,
                "splits": build_outer_splits(
                    len(y),
                    n_splits=OUTER_SPLITS,
                    test_size=BIOMASS_TEST_SIZE,
                    random_state=RANDOM_STATE + 1000 + target_idx,
                ),
            }
        )

    for task in ablation_tasks:
        dataset_name = str(task["dataset_name"])
        target = str(task["target"])
        target_idx = int(task["target_idx"])
        X = np.asarray(task["X"], dtype=float)
        y = np.asarray(task["y"], dtype=float)
        outer_strata = task["outer_strata"]
        splits = task["splits"]
        for split_id, (tr, te) in enumerate(splits, start=1):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            inner_strata = np.asarray(outer_strata)[tr] if outer_strata is not None else None
            split_seed = RANDOM_STATE + 5000 + 100 * target_idx + split_id
            k_auto_preview = ProFPELModel.preview_rate_based_K(
                X_tr,
                y_tr,
                base_learner=PRIMARY_BASE_LEARNER,
            )
            k_auto = int(k_auto_preview["K"])
            print(
                f"  [Ablation] {dataset_name} {target} split {split_id}/{len(splits)} "
                f"K_auto(primary,adaptive)={k_auto} profile={DEFAULT_PROFPEL_KWARGS['profile_metric']}",
                flush=True,
            )

            fit_cache: Dict[str, Dict[str, Any]] = {}
            for variant_name, overrides, family in ablation_variants:
                _append_ablation_row(
                    dataset_name=dataset_name,
                    target=target,
                    split_id=split_id,
                    X_tr=X_tr,
                    X_te=X_te,
                    y_tr=y_tr,
                    y_te=y_te,
                    inner_strata=inner_strata,
                    split_seed=split_seed,
                    variant_name=variant_name,
                    family=family,
                    overrides=overrides,
                    fit_cache=fit_cache,
                )

    return pd.DataFrame(rows)

def summarize_pure_vs_profpel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    part = df.loc[df["Structure"].isin(["Pure", "ProFPEL", "FPPEL"])].copy()
    if part.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    part["Structure"] = part["Structure"].replace({"FPPEL": "ProFPEL"})

    def _spec(key: str) -> Dict[str, Any]:
        return BASE_LEARNER_SPECS.get(str(key), {})

    wide = (
        part.pivot_table(
            index=["Dataset", "Target", "Split", "BaseLearner"],
            columns="Structure",
            values=["R2", "RMSE", "MAE", "Train_Time"],
            aggfunc="mean",
        )
        .sort_index()
    )
    wide.columns = [f"{metric}_{structure}" for metric, structure in wide.columns]
    wide = wide.reset_index()
    for _col in ("Train_Time_Pure", "Train_Time_ProFPEL"):
        if _col not in wide.columns:
            wide[_col] = np.nan
    if "ModelFamily" in part.columns and part["ModelFamily"].notna().any():
        fam_map = part.drop_duplicates("BaseLearner").set_index("BaseLearner")["ModelFamily"].to_dict()
        lab_map = part.drop_duplicates("BaseLearner").set_index("BaseLearner")["ModelFamilyLabel"].to_dict()
        short_map = part.drop_duplicates("BaseLearner").set_index("BaseLearner")["BaseLearnerShort"].to_dict()
        wide["ModelFamily"] = wide["BaseLearner"].map(lambda k: fam_map.get(k) or _spec(k).get("model_family", ""))
        wide["ModelFamilyLabel"] = wide["BaseLearner"].map(
            lambda k: lab_map.get(k) or _spec(k).get("model_family_label", "")
        )
        wide["BaseLearnerShort"] = wide["BaseLearner"].map(lambda k: short_map.get(k) or _spec(k).get("short", k))
    else:
        wide["ModelFamily"] = wide["BaseLearner"].map(lambda k: _spec(k).get("model_family", ""))
        wide["ModelFamilyLabel"] = wide["BaseLearner"].map(lambda k: _spec(k).get("model_family_label", ""))
        wide["BaseLearnerShort"] = wide["BaseLearner"].map(lambda k: _spec(k).get("short", k))
    wide["ModelType"] = wide["BaseLearnerShort"]
    wide["Delta_R2"] = wide["R2_ProFPEL"] - wide["R2_Pure"]
    wide["Delta_RMSE"] = wide["RMSE_Pure"] - wide["RMSE_ProFPEL"]
    wide["Delta_MAE"] = wide["MAE_Pure"] - wide["MAE_ProFPEL"]
    wide["ProFPEL_win"] = (wide["Delta_R2"] > 1e-12).astype(float)
    wide["Delta_Train_Time"] = wide["Train_Time_ProFPEL"] - wide["Train_Time_Pure"]
    wide["Train_Time_ratio"] = np.where(
        wide["Train_Time_Pure"].astype(float) > 1e-12,
        wide["Train_Time_ProFPEL"].astype(float) / wide["Train_Time_Pure"].astype(float),
        np.nan,
    )

    summary = (
        wide.groupby(
            ["Dataset", "Target", "ModelFamily", "ModelFamilyLabel", "BaseLearner", "BaseLearnerShort"],
            dropna=False,
        )
        .agg(
            Pure_B_R2=("R2_Pure", "mean"),
            ProFPEL_B_R2=("R2_ProFPEL", "mean"),
            Delta_R2=("Delta_R2", "mean"),
            WinRate=("ProFPEL_win", "mean"),
            Pure_B_RMSE=("RMSE_Pure", "mean"),
            ProFPEL_B_RMSE=("RMSE_ProFPEL", "mean"),
            Delta_RMSE=("Delta_RMSE", "mean"),
            Pure_B_Train_Time=("Train_Time_Pure", "mean"),
            ProFPEL_B_Train_Time=("Train_Time_ProFPEL", "mean"),
            Delta_Train_Time=("Delta_Train_Time", "mean"),
            Train_Time_ratio=("Train_Time_ratio", "mean"),
            n=("Split", "count"),
        )
        .reset_index()
    )
    overall = (
        wide.groupby(
            ["ModelFamily", "ModelFamilyLabel", "BaseLearner", "BaseLearnerShort"],
            dropna=False,
        )
        .agg(
            Pure_B_R2=("R2_Pure", "mean"),
            ProFPEL_B_R2=("R2_ProFPEL", "mean"),
            Delta_R2=("Delta_R2", "mean"),
            WinRate=("ProFPEL_win", "mean"),
            Pure_B_RMSE=("RMSE_Pure", "mean"),
            ProFPEL_B_RMSE=("RMSE_ProFPEL", "mean"),
            Delta_RMSE=("Delta_RMSE", "mean"),
            Pure_B_Train_Time=("Train_Time_Pure", "mean"),
            ProFPEL_B_Train_Time=("Train_Time_ProFPEL", "mean"),
            Delta_Train_Time=("Delta_Train_Time", "mean"),
            Train_Time_ratio=("Train_Time_ratio", "mean"),
            n=("Split", "count"),
        )
        .reset_index()
    )
    fam_rank = {f: i for i, f in enumerate(MODEL_FAMILY_ORDER)}
    overall["_family_order"] = overall["ModelFamily"].map(lambda x: fam_rank.get(str(x), len(MODEL_FAMILY_ORDER)))
    overall = overall.sort_values(["_family_order", "BaseLearnerShort", "BaseLearner"]).drop(
        columns=["_family_order"]
    )
    overall = overall.reset_index(drop=True)
    return wide, summary, overall


def summarize_primary_vs_strong(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary_label = BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER]["cv_label"]
    primary = df.loc[df["Model"].eq(primary_label), ["Dataset", "Target", "Split", "R2", "RMSE"]].copy()
    primary = primary.rename(columns={"R2": "Primary_R2", "RMSE": "Primary_RMSE"})
    strong = df.loc[
        df["Structure"].isin(["StrongBaseline", "BlackBoxBaseline"]),
        ["Dataset", "Target", "Split", "Model", "R2", "RMSE"],
    ].copy()
    strong = strong.rename(columns={"R2": "Strong_R2", "RMSE": "Strong_RMSE"})
    if primary.empty or strong.empty:
        return pd.DataFrame(), pd.DataFrame()

    paired = strong.merge(primary, on=["Dataset", "Target", "Split"], how="inner")
    paired["Delta_R2_PrimaryMinusStrong"] = paired["Primary_R2"] - paired["Strong_R2"]
    paired["Delta_RMSE_StrongMinusPrimary"] = paired["Strong_RMSE"] - paired["Primary_RMSE"]

    summary = (
        paired.groupby(["Dataset", "Target", "Model"], dropna=False)
        .agg(
            Primary_R2=("Primary_R2", "mean"),
            Strong_R2=("Strong_R2", "mean"),
            Delta_R2_PrimaryMinusStrong=("Delta_R2_PrimaryMinusStrong", "mean"),
            Primary_RMSE=("Primary_RMSE", "mean"),
            Strong_RMSE=("Strong_RMSE", "mean"),
            Delta_RMSE_StrongMinusPrimary=("Delta_RMSE_StrongMinusPrimary", "mean"),
            n=("Split", "count"),
        )
        .reset_index()
    )
    return paired, summary


def summarize_ablation(df_ablation: pd.DataFrame) -> pd.DataFrame:
    if df_ablation.empty:
        return pd.DataFrame()
    return (
        df_ablation.groupby(["Dataset", "Target", "AblationFamily", "AblationVariant"], dropna=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            Train_Time_mean=("Train_Time", "mean"),
            n=("Split", "count"),
        )
        .reset_index()
    )


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray([np.nan if v is None else float(v) for v in p_values], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return out.tolist()
    order = valid[np.argsort(p[valid])]
    m = len(order)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[rank] = running
    for rank, idx in enumerate(order):
        out[idx] = adjusted[rank]
    return out.tolist()


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = RANDOM_STATE,
    n_boot: int = 5000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(int(n_boot), arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def paired_delta_stats(values: Sequence[float], label: str, *, seed: int = RANDOM_STATE) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "Comparison": label,
            "n": 0,
            "DeltaMean": np.nan,
            "DeltaMedian": np.nan,
            "BootstrapCI95_L": np.nan,
            "BootstrapCI95_U": np.nan,
            "PositiveRate": np.nan,
            "Win": 0,
            "Tie": 0,
            "Loss": 0,
            "CohenDz": np.nan,
            "Wilcoxon_p": np.nan,
            "TTest_p": np.nan,
        }
    ci_l, ci_u = bootstrap_ci(arr, seed=seed)
    std = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    wilcoxon_p = np.nan
    ttest_p = np.nan
    if n > 1:
        if HAS_SCIPY_STATS:
            try:
                if np.any(np.abs(arr) > 1e-12):
                    wilcoxon_p = float(scipy_stats.wilcoxon(arr, zero_method="wilcox").pvalue)
            except Exception:
                wilcoxon_p = np.nan
            try:
                ttest_p = float(scipy_stats.ttest_1samp(arr, popmean=0.0).pvalue)
            except Exception:
                ttest_p = np.nan
    return {
        "Comparison": label,
        "n": n,
        "DeltaMean": float(np.mean(arr)),
        "DeltaMedian": float(np.median(arr)),
        "BootstrapCI95_L": ci_l,
        "BootstrapCI95_U": ci_u,
        "PositiveRate": float(np.mean(arr > 0.0)),
        "Win": int(np.sum(arr > 1e-12)),
        "Tie": int(np.sum(np.abs(arr) <= 1e-12)),
        "Loss": int(np.sum(arr < -1e-12)),
        "CohenDz": float(np.mean(arr) / std) if n > 1 and std > 1e-12 else np.nan,
        "Wilcoxon_p": wilcoxon_p,
        "TTest_p": ttest_p,
    }


ABLATION_REFERENCE_BY_FAMILY: Dict[str, str] = {
    "region_structure": "adaptive_K",
    "profile_information": "value_slope",
    "partition_gate": "soft_gate",
    "feature_order": "target_corr_order",
}


def paired_ablation_deltas(df_ablation: pd.DataFrame) -> pd.DataFrame:
    """Family-wise ablation deltas as reference minus variant on matched dataset/target/split."""
    if df_ablation.empty:
        return pd.DataFrame()
    need = {"Dataset", "Target", "Split", "AblationFamily", "AblationVariant", "R2", "RMSE", "MAE"}
    if not need.issubset(df_ablation.columns):
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for family, ref_variant in ABLATION_REFERENCE_BY_FAMILY.items():
        fam = df_ablation.loc[df_ablation["AblationFamily"].eq(family)].copy()
        if fam.empty:
            continue
        ref = fam.loc[
            fam["AblationVariant"].eq(ref_variant),
            ["Dataset", "Target", "Split", "R2", "RMSE", "MAE"],
        ].rename(columns={"R2": "Reference_R2", "RMSE": "Reference_RMSE", "MAE": "Reference_MAE"})
        others = fam.loc[~fam["AblationVariant"].eq(ref_variant)].copy()
        if ref.empty or others.empty:
            continue
        paired = others.merge(ref, on=["Dataset", "Target", "Split"], how="inner")
        if paired.empty:
            continue
        paired["ReferenceVariant"] = ref_variant
        paired["Delta_R2_ReferenceMinusVariant"] = paired["Reference_R2"] - paired["R2"]
        paired["Delta_RMSE_VariantMinusReference"] = paired["RMSE"] - paired["Reference_RMSE"]
        paired["Delta_MAE_VariantMinusReference"] = paired["MAE"] - paired["Reference_MAE"]
        rows.append(paired)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    sort_cols = ["Dataset", "Target", "AblationFamily", "AblationVariant", "Split"]
    return out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def ablation_stat_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired reference-minus-ablation R2 tests; Holm adjusted within each ablation family."""
    if paired.empty or "Delta_R2_ReferenceMinusVariant" not in paired.columns:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for family, fam in paired.groupby("AblationFamily", dropna=False):
        fam_rows: List[Dict[str, Any]] = []
        for (ds, tgt, variant), part in fam.groupby(["Dataset", "Target", "AblationVariant"], dropna=False):
            ref = str(part["ReferenceVariant"].iloc[0]) if "ReferenceVariant" in part.columns and len(part) else ""
            st = paired_delta_stats(
                part["Delta_R2_ReferenceMinusVariant"].values,
                f"{ref} minus {variant} | {ds} | {tgt} | {family}",
            )
            fam_rows.append(
                {
                    "Dataset": ds,
                    "Target": tgt,
                    "AblationFamily": family,
                    "ReferenceVariant": ref,
                    "AblationVariant": variant,
                    **st,
                }
            )
        wp = [float(r.get("Wilcoxon_p", np.nan)) for r in fam_rows]
        tp = [float(r.get("TTest_p", np.nan)) for r in fam_rows]
        wh = holm_adjust(wp)
        th = holm_adjust(tp)
        for i, r in enumerate(fam_rows):
            r["Wilcoxon_p_Holm"] = wh[i] if i < len(wh) else np.nan
            r["TTest_p_Holm"] = th[i] if i < len(th) else np.nan
        rows.extend(fam_rows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["AblationFamily", "Dataset", "Target", "AblationVariant"],
            kind="mergesort",
        ).reset_index(drop=True)
    return out


def pure_vs_profpel_stat_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired FPEL-minus-Pure R2 deltas; Holm correction within each (Dataset, Target) across base learners."""
    if paired.empty:
        return pd.DataFrame()
    need = {"Dataset", "Target", "BaseLearner", "Delta_R2"}
    if not need.issubset(paired.columns):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (ds, tgt), g_task in paired.groupby(["Dataset", "Target"], dropna=False):
        task_rows: List[Dict[str, Any]] = []
        for base, part in g_task.groupby("BaseLearner", dropna=False):
            st = paired_delta_stats(
                part["Delta_R2"].values,
                f"FPEL vs Pure | {ds} | {tgt} | {base}",
            )
            task_rows.append({"Dataset": ds, "Target": tgt, "BaseLearner": base, **st})
        if not task_rows:
            continue
        wp = [float(r.get("Wilcoxon_p", np.nan)) for r in task_rows]
        tp = [float(r.get("TTest_p", np.nan)) for r in task_rows]
        wh = holm_adjust(wp)
        th = holm_adjust(tp)
        for i, r in enumerate(task_rows):
            r["Wilcoxon_p_Holm"] = wh[i] if i < len(wh) else np.nan
            r["TTest_p_Holm"] = th[i] if i < len(th) else np.nan
        rows.extend(task_rows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dataset", "Target", "BaseLearner"], kind="mergesort").reset_index(drop=True)
    return out


def pure_vs_profpel_train_time_stat_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired **FPEL train time minus Pure** with Holm **within each (Dataset, Target)** across base learners."""
    if paired.empty or "Delta_Train_Time" not in paired.columns:
        return pd.DataFrame()
    need = {"Dataset", "Target", "BaseLearner", "Delta_Train_Time"}
    if not need.issubset(paired.columns):
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (ds, tgt), g_task in paired.groupby(["Dataset", "Target"], dropna=False):
        task_rows: List[Dict[str, Any]] = []
        for base, part in g_task.groupby("BaseLearner", dropna=False):
            st = paired_delta_stats(
                part["Delta_Train_Time"].values,
                f"FPEL minus Pure train time s | {ds} | {tgt} | {base}",
            )
            task_rows.append({"Dataset": ds, "Target": tgt, "BaseLearner": base, **st})
        if not task_rows:
            continue
        wp = [float(r.get("Wilcoxon_p", np.nan)) for r in task_rows]
        tp = [float(r.get("TTest_p", np.nan)) for r in task_rows]
        wh = holm_adjust(wp)
        th = holm_adjust(tp)
        for i, r in enumerate(task_rows):
            r["Wilcoxon_p_Holm"] = wh[i] if i < len(wh) else np.nan
            r["TTest_p_Holm"] = th[i] if i < len(th) else np.nan
        rows.extend(task_rows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dataset", "Target", "BaseLearner"], kind="mergesort").reset_index(drop=True)
    return out


def primary_vs_strong_stat_tests(paired: pd.DataFrame) -> pd.DataFrame:
    """Paired primary-minus-strong-baseline R2 deltas; Holm within each (Dataset, Target) across strong models."""
    if paired.empty or "Delta_R2_PrimaryMinusStrong" not in paired.columns:
        return pd.DataFrame()
    need = {"Dataset", "Target", "Model", "Delta_R2_PrimaryMinusStrong"}
    if not need.issubset(paired.columns):
        return pd.DataFrame()
    primary_label = BASE_LEARNER_SPECS[PRIMARY_BASE_LEARNER]["cv_label"]
    rows: List[Dict[str, Any]] = []
    for (ds, tgt), g_task in paired.groupby(["Dataset", "Target"], dropna=False):
        task_rows: List[Dict[str, Any]] = []
        for model, part in g_task.groupby("Model", dropna=False):
            st = paired_delta_stats(
                part["Delta_R2_PrimaryMinusStrong"].values,
                f"{primary_label} vs {model} | {ds} | {tgt}",
            )
            task_rows.append({"Dataset": ds, "Target": tgt, "Model": model, **st})
        if not task_rows:
            continue
        wp = [float(r.get("Wilcoxon_p", np.nan)) for r in task_rows]
        tp = [float(r.get("TTest_p", np.nan)) for r in task_rows]
        wh = holm_adjust(wp)
        th = holm_adjust(tp)
        for i, r in enumerate(task_rows):
            r["Wilcoxon_p_Holm"] = wh[i] if i < len(wh) else np.nan
            r["TTest_p_Holm"] = th[i] if i < len(th) else np.nan
        rows.extend(task_rows)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Dataset", "Target", "Model"], kind="mergesort").reset_index(drop=True)
    return out


def average_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    usable = df.copy()
    usable["Rank"] = usable.groupby(["Dataset", "Target", "Split"])["R2"].rank(ascending=False, method="average")
    return (
        usable.groupby(["Dataset", "Target", "Model"], dropna=False)
        .agg(
            AverageRank=("Rank", "mean"),
            MeanR2=("R2", "mean"),
            MeanRMSE=("RMSE", "mean"),
            n=("Split", "count"),
        )
        .reset_index()
        .sort_values(["Dataset", "Target", "AverageRank", "MeanR2"], ascending=[True, True, True, False])
    )


def _save_figure(fig: Any, stem: str, title: str) -> List[Dict[str, Any]]:
    ensure_dir(FIG_DIR)
    rows: List[Dict[str, Any]] = []
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        rows.append({"Figure": stem, "Title": title, "Format": ext, "Path": path})
    return rows


def export_publication_figures(
    *,
    pv_pairs: pd.DataFrame,
    pv_summary: pd.DataFrame,
    pv_overall: pd.DataFrame,
    strong_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    ablation_pairs: pd.DataFrame,
    avg_rank: pd.DataFrame,
) -> pd.DataFrame:
    """Export compact publication-ready PNG/PDF figures from final result tables."""
    ensure_dir(FIG_DIR)
    if not HAS_MATPLOTLIB or plt is None:
        return pd.DataFrame(
            [{"Figure": "not_generated", "Title": "matplotlib is unavailable", "Format": "", "Path": ""}]
        )

    figure_rows: List[Dict[str, Any]] = []
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )

    if not pv_overall.empty and {"BaseLearnerShort", "Delta_R2"}.issubset(pv_overall.columns):
        plot_df = pv_overall.copy().sort_values("Delta_R2")
        fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(plot_df))))
        colors = ["#b44b4b" if v < 0 else "#2f6f9f" for v in plot_df["Delta_R2"].astype(float)]
        ax.barh(plot_df["BaseLearnerShort"].astype(str), plot_df["Delta_R2"].astype(float), color=colors)
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_xlabel("Mean held-out R2 gain (FPEL-B minus Pure-B)")
        ax.set_ylabel("Base learner")
        ax.set_title("Overall FPEL gain by white-box base learner")
        figure_rows.extend(_save_figure(fig, "fig1_overall_fpel_gain_by_learner", ax.get_title()))
        plt.close(fig)

    if not pv_summary.empty and {"Dataset", "Target", "BaseLearner", "Delta_R2"}.issubset(pv_summary.columns):
        primary = pv_summary.loc[pv_summary["BaseLearner"].eq(PRIMARY_BASE_LEARNER)].copy()
        if not primary.empty:
            primary["Task"] = primary["Dataset"].astype(str) + " | " + primary["Target"].astype(str)
            primary = primary.sort_values(["Dataset", "Target"])
            fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.28 * len(primary))))
            colors = ["#b44b4b" if v < 0 else "#2f6f9f" for v in primary["Delta_R2"].astype(float)]
            ax.barh(primary["Task"], primary["Delta_R2"].astype(float), color=colors)
            ax.axvline(0.0, color="0.25", linewidth=0.8)
            ax.set_xlabel("Mean held-out R2 gain (FPEL-Ridge-Q minus Pure Ridge-Q)")
            ax.set_ylabel("Dataset | target")
            ax.set_title("Primary model gain across all tasks")
            figure_rows.extend(_save_figure(fig, "fig2_primary_gain_by_task", ax.get_title()))
            plt.close(fig)

    if not avg_rank.empty and {"Model", "AverageRank", "MeanR2"}.issubset(avg_rank.columns):
        rank_df = (
            avg_rank.groupby("Model", dropna=False)
            .agg(AverageRank=("AverageRank", "mean"), MeanR2=("MeanR2", "mean"))
            .reset_index()
            .sort_values(["AverageRank", "MeanR2"], ascending=[True, False])
            .head(18)
        )
        fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.32 * len(rank_df))))
        ax.barh(rank_df["Model"].astype(str)[::-1], rank_df["AverageRank"].astype(float)[::-1], color="#4f7f52")
        ax.set_xlabel("Average rank across dataset-target-split evaluations (lower is better)")
        ax.set_ylabel("Model")
        ax.set_title("Model ranking across all prediction tasks")
        figure_rows.extend(_save_figure(fig, "fig3_average_rank_all_models", ax.get_title()))
        plt.close(fig)

    if not ablation_pairs.empty and {"AblationFamily", "AblationVariant", "Delta_R2_ReferenceMinusVariant"}.issubset(
        ablation_pairs.columns
    ):
        abl = (
            ablation_pairs.groupby(["AblationFamily", "AblationVariant", "ReferenceVariant"], dropna=False)
            .agg(
                DeltaMean=("Delta_R2_ReferenceMinusVariant", "mean"),
                DeltaStd=("Delta_R2_ReferenceMinusVariant", "std"),
                n=("Split", "count"),
            )
            .reset_index()
        )
        abl["Label"] = abl["AblationFamily"].astype(str) + ": " + abl["AblationVariant"].astype(str)
        abl = abl.sort_values(["AblationFamily", "DeltaMean"])
        fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.32 * len(abl))))
        xerr = (abl["DeltaStd"].fillna(0.0) / np.sqrt(abl["n"].clip(lower=1))).astype(float)
        colors = ["#b44b4b" if v < 0 else "#6b5b95" for v in abl["DeltaMean"].astype(float)]
        ax.barh(abl["Label"], abl["DeltaMean"].astype(float), xerr=xerr, color=colors, ecolor="0.25")
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_xlabel("Mean paired R2 drop vs family reference (reference minus variant)")
        ax.set_ylabel("Ablation")
        ax.set_title("Mechanism ablation of the primary FPEL model")
        figure_rows.extend(_save_figure(fig, "fig4_primary_ablation_deltas", ax.get_title()))
        plt.close(fig)

    if not pv_overall.empty and {"BaseLearnerShort", "Train_Time_ratio"}.issubset(pv_overall.columns):
        time_df = pv_overall.copy()
        time_df = time_df[np.isfinite(time_df["Train_Time_ratio"].astype(float))].sort_values("Train_Time_ratio")
        if not time_df.empty:
            fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(time_df))))
            ax.barh(time_df["BaseLearnerShort"].astype(str), time_df["Train_Time_ratio"].astype(float), color="#7b6f4f")
            ax.axvline(1.0, color="0.25", linewidth=0.8)
            ax.set_xlabel("Training-time ratio (FPEL-B / Pure-B)")
            ax.set_ylabel("Base learner")
            ax.set_title("Computational overhead of FPEL")
            figure_rows.extend(_save_figure(fig, "fig5_training_time_ratio", ax.get_title()))
            plt.close(fig)

    return pd.DataFrame(figure_rows)


def write_protocol_note() -> str:
    ensure_dir(OUT_DIR)
    path = os.path.join(OUT_DIR, "fpel_protocol.txt")
    lines = [
        "Feature-Profile Enhanced Learning (FPEL) experimental protocol",
        f"quick_mode = {QUICK_MODE}",
        f"outer_splits = {OUTER_SPLITS}",
        f"inner_cv_folds = {INNER_CV_FOLDS}",
        f"parallel_n_jobs = {PARALLEL_N_JOBS}",
        f"parallel_backend = {PARALLEL_BACKEND}",
        "selection_protocol = FPEL-B repeats the same Pure-equivalent white-box inner CV, fixes the winner, then runs fold-local static FPEL fits for stacked OOF and a full-training refit; Train_Time per Structure is wall-clock for that row's full train+selection block",
        "fpel_partition_protocol = profile gates from K-means (or mean) centers and bandwidth; one inner-CV stack per fold fit and one full-data refit; gates are not updated in an outer loop",
        "selection_note = validation folds do not participate in base-learner hyperparameter selection or FPEL gate/K selection (Structure=ProFPEL rows)",
        "whitebox_selection_objective = base-learner hyperparameters maximize Pure inner-CV stacked OOF R2; the same winning vector is passed to FPEL runs (Structure=ProFPEL; not re-optimized for stacked FPEL OOF)",
        "partition_oof_note = stacked OOF uses fixed gates on each inner fold; final local experts use the same gate construction on all training rows",
        "task_scope = regression only in this version; classification is a future extension",
        f"biomass_test_size = {BIOMASS_TEST_SIZE}",
        "benchmark_data = BiomassGasification, HHV, and CoGasification workbooks",
        "default_n_regions = adaptive Gaussian-mixture BIC on feature-profile matrix P (k_selection=adaptive_gmm_bic; k_min=1; K capped by min(n-1, floor(n/p)) with p=input feature count unless k_max tightens further; min_region_samples unset unless set); k_selection=gmm_bic uses bounded-grid BIC with the same cap; explicit n_regions fixes K for ablations.",
        "auto_K_note = inner-fold static FPEL fits choose K on each fold's training subset; the full-data FPEL refit chooses K on all training rows of that outer split; K may therefore differ across inner folds",
        f"default_profile_metric = {DEFAULT_PROFPEL_KWARGS['profile_metric']}",
        "feature_order_note = default feature_order target_corr ranks features by absolute Pearson correlation with y on the training material used for that fit (inner-CV folds use inner-train rows only); original order is label-agnostic",
        "structure_rule = learn K soft feature-profile regions and train every local expert on all samples with region-specific weights",
        "soft_gate_rule = predictions are blended by continuous softmax gates in feature-profile space",
        "ablation_grids = region_structure: adaptive_K, global_only_K1; profile_information: value_slope, value_only, slope_only, shuffled_profile; partition_gate: soft_gate, hard_partition; feature_order: target_corr_order, original_order",
        "k_ablation_note = global_only_K1 uses explicit n_regions=1; adaptive_K uses k_selection=adaptive_gmm_bic",
        "ablation_reference_note = paired ablation deltas use family references: adaptive_K, value_slope, soft_gate, and target_corr_order; positive reference-minus-variant R2 means the reference retained performance",
        "supervised_fallback = FPPEL_SUPERVISED_FALLBACK_TOL or select_profpel(supervised_fallback_tol=...); when tol is set and stacked FPEL OOF R2 <= whitebox_sel_R2 + tol, the returned estimator is Pure-B on locked params without a full-data FPEL refit; when tol is unset, the full-data FPEL refit always runs",
        "statistical_tests = paired deltas on R2 and train-time (s): bootstrap 95% CI on outer-split deltas, Wilcoxon signed-rank and one-sample t vs 0, win/tie/loss; Holm adjusts p-values within each (Dataset, Target) family across base learners (Pure vs FPEL, Structure=ProFPEL) or across strong-baseline models (primary vs strong); comparisons across different tasks are not jointly adjusted",
        "excel_summary_sheets = all_results_raw plus one raw sheet per dataset, ablation_raw; pure_vs_fpel_all, pure_vs_fpel_by_type, pure_vs_fpel_sum, pure_vs_fpel_stats, pure_vs_fpel_time_stats; primary_vs_strong, strong_summary, strong_stats (non-empty when Structure is StrongBaseline or BlackBoxBaseline); ablation_summary, ablation_pairs, ablation_stats; primary_coeffs, primary_regions, average_rank, figure_index",
        f"figure_outputs = publication PNG/PDF figures are written under {FIG_DIR}",
        "pure_vs_fpel_by_type_note = exploratory macro-average over base learners within ModelFamily; descriptive context alongside split-level primary endpoints",
        "ProFPEL_LocalWeightsAccepted = 1 when sample_weight is applied on every final local expert fit for Structure=ProFPEL (FPEL) rows; NaN on Pure rows",
        "PROFPEL_STRICT_SAMPLE_WEIGHT or FPPEL_STRICT_SAMPLE_WEIGHT = 1/true enforces sample_weight on every local expert fit (RuntimeError otherwise)",
        f"primary_base_learner = {PRIMARY_BASE_LEARNER}",
        "learner_set = FPPEL_LEARNER_SET selects WHITEBOX_PROFPEL_LEARNERS for Pure-B vs FPEL-B (Structure=ProFPEL); BLACK_BOX_BASELINE_LEARNERS always run as Structure=BlackBoxBaseline with standalone inner-CV tuning on BLACK_BOX_TUNING_GRIDS",
        f"whitebox_profpel_learners = {list(WHITEBOX_PROFPEL_LEARNERS)}",
        f"black_box_baselines = {list(active_black_box_baselines())}",
        f"main_base_learners (white-box FPEL pool) = {list(active_main_base_learners())}",
        "datasets = BiomassGasification; HHV with Ash (dry), C%, H%, O%, N%, S% -> HHV; CoGasification with target-wise complete-case modeling for Syngas_yield, H2, CO2, CH4, CO, and Syngas_LHV",
        "",
        "Model-type matrix (white-box Pure-B vs FPEL-B; same inner-CV grid per B):",
    ]
    for fam in MODEL_FAMILY_ORDER:
        fam_keys = [k for k in WHITEBOX_PROFPEL_LEARNERS if BASE_LEARNER_SPECS.get(k, {}).get("model_family") == fam]
        if not fam_keys:
            continue
        label = BASE_LEARNER_SPECS[fam_keys[0]].get("model_family_label", fam)
        lines.append(f"  [{fam}] {label}")
        for key in fam_keys:
            sp = BASE_LEARNER_SPECS[key]
            dep = sp.get("optional_dependency")
            avail = is_base_learner_available(key)
            dep_note = f" [optional: {dep}, installed={avail}]" if dep else ""
            lines.append(f"    - {sp['pure_label']} vs {sp['cv_label']}  ({key}){dep_note}")
    lines.append("")
    lines.append("Black-box baselines (inner-CV tuned; Structure=BlackBoxBaseline; standalone predictors):")
    for key in active_black_box_baselines():
        lines.append(f"  - {BLACK_BOX_BASELINE_LABELS[key]}: grid = {BLACK_BOX_TUNING_GRIDS[key]}")
    lines.append("")
    lines.append("Runtime environment:")
    for env_line in collect_runtime_environment_lines():
        lines.append(f"  {env_line}")
    lines.append("")
    lines.append("White-box base-learner search spaces (active white-box set):")
    for key in active_main_base_learners():
        lines.append(f"  - {key}: {BASE_LEARNER_SPECS[key]['grid']}")
    lines.append("")
    lines.append("Data source:")
    lines.append(f"  - Biomass workbook: {BIOMASS_WORKBOOK_NAME!r} (see FPPEL_BIOMASS_PATH / locate_biomass_workbook)")
    lines.append(f"  - HHV workbook: {HHV_WORKBOOK_NAME!r} (see FPPEL_HHV_PATH / load_hhv_data)")
    lines.append(
        f"  - Co-gasification workbook: {CO_GAS_WORKBOOK_NAME!r} "
        "(see FPPEL_CO_GAS_PATH / load_co_gasification_data; target-wise complete-case rows)"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def export_results(
    results_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    coeff_df: pd.DataFrame,
    region_df: pd.DataFrame,
) -> str:
    ensure_dir(OUT_DIR)
    combined = results_df.copy()
    pv_pairs, pv_summary, pv_overall = summarize_pure_vs_profpel(combined)
    pv_family = pd.DataFrame()
    if not pv_overall.empty:
        fam_rank = {f: i for i, f in enumerate(MODEL_FAMILY_ORDER)}
        pv_family = (
            pv_overall.groupby(["ModelFamily", "ModelFamilyLabel"], dropna=False)
            .agg(
                Pure_B_R2=("Pure_B_R2", "mean"),
                ProFPEL_B_R2=("ProFPEL_B_R2", "mean"),
                Delta_R2=("Delta_R2", "mean"),
                WinRate=("WinRate", "mean"),
                Pure_B_Train_Time=("Pure_B_Train_Time", "mean"),
                ProFPEL_B_Train_Time=("ProFPEL_B_Train_Time", "mean"),
                Delta_Train_Time=("Delta_Train_Time", "mean"),
                Train_Time_ratio=("Train_Time_ratio", "mean"),
                n_tasks=("n", "sum"),
                n_learners=("BaseLearner", "count"),
            )
            .reset_index()
        )
        pv_family["_family_order"] = pv_family["ModelFamily"].map(
            lambda x: fam_rank.get(str(x), len(MODEL_FAMILY_ORDER))
        )
        pv_family = pv_family.sort_values("_family_order").drop(columns=["_family_order"]).reset_index(drop=True)
    strong_pairs, strong_summary = summarize_primary_vs_strong(combined)
    pv_stats = pure_vs_profpel_stat_tests(pv_pairs)
    pv_time_stats = pure_vs_profpel_train_time_stat_tests(pv_pairs)
    strong_stats = primary_vs_strong_stat_tests(strong_pairs)
    ablation_summary = summarize_ablation(ablation_df)
    ablation_pairs = paired_ablation_deltas(ablation_df)
    ablation_stats = ablation_stat_tests(ablation_pairs)
    avg_rank = average_rank_table(combined)
    figure_index = export_publication_figures(
        pv_pairs=pv_pairs,
        pv_summary=pv_summary,
        pv_overall=pv_overall,
        strong_summary=strong_summary,
        ablation_summary=ablation_summary,
        ablation_pairs=ablation_pairs,
        avg_rank=avg_rank,
    )

    out_path = os.path.join(OUT_DIR, "fpel_experiments.xlsx")
    with pd.ExcelWriter(out_path) as writer:
        results_df.to_excel(writer, sheet_name="all_results_raw", index=False)
        for dataset_name, dataset_part in results_df.groupby("Dataset", dropna=False):
            sheet = safe_name(str(dataset_name))[:25] + "_raw"
            dataset_part.to_excel(writer, sheet_name=sheet[:31], index=False)
        ablation_df.to_excel(writer, sheet_name="ablation_raw", index=False)
        pv_pairs.to_excel(writer, sheet_name="pure_vs_fpel_pairs", index=False)
        pv_summary.to_excel(writer, sheet_name="pure_vs_fpel_sum", index=False)
        pv_overall.to_excel(writer, sheet_name="pure_vs_fpel_all", index=False)
        pv_family.to_excel(writer, sheet_name="pure_vs_fpel_by_type", index=False)
        pv_stats.to_excel(writer, sheet_name="pure_vs_fpel_stats", index=False)
        pv_time_stats.to_excel(writer, sheet_name="pure_vs_fpel_time_stats", index=False)
        strong_pairs.to_excel(writer, sheet_name="primary_vs_strong", index=False)
        strong_summary.to_excel(writer, sheet_name="strong_summary", index=False)
        strong_stats.to_excel(writer, sheet_name="strong_stats", index=False)
        ablation_summary.to_excel(writer, sheet_name="ablation_summary", index=False)
        ablation_pairs.to_excel(writer, sheet_name="ablation_pairs", index=False)
        ablation_stats.to_excel(writer, sheet_name="ablation_stats", index=False)
        coeff_df.to_excel(writer, sheet_name="primary_coeffs", index=False)
        region_df.to_excel(writer, sheet_name="primary_regions", index=False)
        avg_rank.to_excel(writer, sheet_name="average_rank", index=False)
        figure_index.to_excel(writer, sheet_name="figure_index", index=False)
    return out_path


def main() -> None:
    ensure_dir(OUT_DIR)
    set_seed(RANDOM_STATE)

    t0 = time.time()
    print(
        f"[FPEL] main() started | QUICK_MODE={QUICK_MODE} parallel_backend={PARALLEL_BACKEND!r} "
        f"n_jobs={PARALLEL_N_JOBS}",
        flush=True,
    )

    biomass_df, biomass_coeff_df, biomass_region_df = run_biomass_experiment()
    hhv_df, hhv_coeff_df, hhv_region_df = run_hhv_experiment()
    co_gas_df, co_gas_coeff_df, co_gas_region_df = run_co_gasification_experiment()
    results_df = pd.concat([biomass_df, hhv_df, co_gas_df], ignore_index=True)
    coeff_df = pd.concat([biomass_coeff_df, hhv_coeff_df, co_gas_coeff_df], ignore_index=True)
    region_df = pd.concat([biomass_region_df, hhv_region_df, co_gas_region_df], ignore_index=True)
    ablation_df = run_primary_ablation()

    workbook = export_results(
        results_df=results_df,
        ablation_df=ablation_df,
        coeff_df=coeff_df,
        region_df=region_df,
    )
    protocol_path = write_protocol_note()

    combined = results_df.copy()
    _, pv_summary, pv_overall = summarize_pure_vs_profpel(combined)

    print("\n" + "=" * 78)
    print("Experiment complete")
    print("=" * 78)
    if not pv_overall.empty:
        print("[Overall Pure-B vs FPEL-B]")
        print(pv_overall.to_string(index=False))
        tcols = [
            "ModelFamilyLabel",
            "BaseLearnerShort",
            "Pure_B_Train_Time",
            "ProFPEL_B_Train_Time",
            "Delta_Train_Time",
            "Train_Time_ratio",
        ]
        if all(c in pv_overall.columns for c in tcols):
            print("\n[Mean train time (s) over splits: Pure-B vs FPEL-B]")
            print(pv_overall[tcols].to_string(index=False))
    print(f"\nWorkbook : {workbook}")
    print(f"Protocol : {protocol_path}")
    print(f"Elapsed  : {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
