"""
=============================================================================
config.py — Global Configuration, Seeds, and Hyperparameters
=============================================================================
Purpose:
    Single source of truth for every tunable parameter in the pipeline.
    Centralising configuration here ensures:
      1. Full reproducibility — anyone cloning the repo gets identical results.
      2. Hyperparameter transparency — required for journal publication.
      3. Easy cloud override — AWS SageMaker / EC2 can inject values via
         environment variables without modifying source code.

Academic reference:
    Reproducibility in ML research — Pineau et al., "Improving Reproducibility
    in Machine Learning Research", JMLR 2021.
=============================================================================
"""

import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# 1.  RANDOM SEEDS
#     Setting all seeds guarantees byte-identical results across runs.
#     The same seed must be passed to every stochastic component.
# ---------------------------------------------------------------------------

GLOBAL_SEED: int = 42          # Master seed — propagated everywhere

random.seed(GLOBAL_SEED)       # Python built-in RNG
np.random.seed(GLOBAL_SEED)    # NumPy RNG (used by Scikit-Learn internally)

# TensorFlow / PyTorch seeds are set lazily inside the model modules because
# importing them here would force GPU initialisation on import.


# ---------------------------------------------------------------------------
# 2.  PATHS
#     Override via environment variables for cloud execution, e.g.:
#       export DATA_DIR=/mnt/s3-mount/cicids2017
# ---------------------------------------------------------------------------

# Root of the project (one level above this file)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR: str = os.environ.get(
    "DATA_DIR",
    os.path.join(PROJECT_ROOT, "data", "raw"),
)

PROCESSED_DIR: str = os.environ.get(
    "PROCESSED_DIR",
    os.path.join(PROJECT_ROOT, "data", "processed"),
)

ARTIFACT_DIR: str = os.environ.get(
    "ARTIFACT_DIR",
    os.path.join(PROJECT_ROOT, "artifacts"),
)

# Sub-directories created at runtime
MODEL_DIR: str  = os.path.join(ARTIFACT_DIR, "models")
REPORT_DIR: str = os.path.join(ARTIFACT_DIR, "reports")
PLOT_DIR: str   = os.path.join(ARTIFACT_DIR, "plots")


# ---------------------------------------------------------------------------
# 3.  DATASET CONFIGURATION
#     CICIDS2017: Canadian Institute for Cybersecurity Intrusion Detection
#     Dataset 2017.  Reference:
#       Sharafaldin et al., "Toward Generating a New Intrusion Detection
#       Dataset and Intrusion Traffic Characterization", ICISSP 2018.
#
#     The dataset ships as multiple CSV files (one per capture day).
#     Each file shares the same 79 network-flow features extracted by
#     CICFlowMeter.  The label column is " Label" (note the leading space).
# ---------------------------------------------------------------------------

# Glob pattern that matches all CICIDS2017 day CSVs
DATASET_GLOB: str = os.path.join(DATA_DIR, "*.csv")

# Raw label column name in CICIDS2017 (CICFlowMeter adds a leading space)
RAW_LABEL_COL: str = " Label"

# After normalisation the column is renamed to this
LABEL_COL: str = "label"

# Value assigned to benign traffic in the binary label
BENIGN_LABEL: str = "BENIGN"

# Whether to use a chronological (time-series) split instead of random.
# Set True when the dataset includes a timestamp column — this simulates
# zero-day detection: the model is trained on Monday–Thursday traffic and
# evaluated on Friday traffic.
CHRONOLOGICAL_SPLIT: bool = True
TIMESTAMP_COL: str = " Timestamp"   # Raw column name in CICIDS2017

# Proportion of data reserved for validation and test sets
VAL_SIZE: float  = 0.15   # 15 % of the full dataset
TEST_SIZE: float = 0.15   # 15 % of the full dataset
# Training set = 1 - VAL_SIZE - TEST_SIZE = 70 %


# ---------------------------------------------------------------------------
# 4.  PREPROCESSING HYPERPARAMETERS
# ---------------------------------------------------------------------------

# Features to drop before modelling (leakage-prone or non-informative)
COLS_TO_DROP: list[str] = [
    " Flow ID",         # Unique per connection — causes data leakage
    " Source IP",       # Raw IPs overfit to lab topology
    " Destination IP",
    " Source Port",     # High-cardinality; represented better by features
    " Destination Port",
    TIMESTAMP_COL,      # After ordering the split; keep no temporal leakage
]

# Strategy for imputing missing / infinite values
IMPUTATION_STRATEGY: str = "median"   # Robust to outliers — preferred for IDS

# Scaler choice: "standard" (zero mean, unit variance) or "minmax" [0, 1]
SCALER: str = "standard"

# SMOTE (Synthetic Minority Over-sampling Technique) parameters
#   Reference: Chawla et al., "SMOTE: Synthetic Minority Over-sampling
#   Technique", JAIR 2002.
USE_SMOTE: bool = True     # Apply only to the training fold, NEVER to test
SMOTE_K_NEIGHBORS: int = 5 # Number of nearest neighbours used by SMOTE


# ---------------------------------------------------------------------------
# 5.  RANDOM FOREST HYPERPARAMETERS
#     Baseline model — interpretable, strong performance on tabular IDS data.
#     Reference: Breiman, "Random Forests", Machine Learning, 2001.
# ---------------------------------------------------------------------------

RF_PARAMS: dict = {
    "n_estimators": 200,        # Number of trees; more = better but slower
    "max_depth": None,          # Grow until leaves are pure (regularise via min_samples)
    "min_samples_split": 5,     # Minimum samples required to split a node
    "min_samples_leaf": 2,      # Minimum samples in any leaf node
    "class_weight": "balanced", # Compensates for class imbalance automatically
    "n_jobs": -1,               # Use all CPU cores
    "random_state": GLOBAL_SEED,
    "verbose": 0,
}


# ---------------------------------------------------------------------------
# 6.  XGBOOST HYPERPARAMETERS
#     High-performance gradient-boosted trees; state-of-the-art on CICIDS2017.
#     Reference: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System",
#     KDD 2016.
# ---------------------------------------------------------------------------

XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,        # Shrinkage — reduces overfitting
    "subsample": 0.8,             # Row subsampling per tree
    "colsample_bytree": 0.8,      # Feature subsampling per tree
    "reg_alpha": 0.1,             # L1 regularisation
    "reg_lambda": 1.0,            # L2 regularisation
    "scale_pos_weight": 1,        # Adjusted dynamically if USE_SMOTE=False
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_jobs": -1,
    "random_state": GLOBAL_SEED,
    "verbosity": 0,
}

# Early stopping patience (measured in XGBoost rounds on the validation set)
XGB_EARLY_STOPPING_ROUNDS: int = 20


# ---------------------------------------------------------------------------
# 7.  DEEP LEARNING HYPERPARAMETERS  (TensorFlow / Keras)
#     A feed-forward neural network used as the third model in the ensemble.
#     Reference: Ring et al., "Flow-based Network Traffic Generation using
#     Deep Learning", Computers & Security, 2019.
# ---------------------------------------------------------------------------

DL_PARAMS: dict = {
    "hidden_units": [256, 128, 64],  # Neurons per hidden layer
    "dropout_rate": 0.3,             # Dropout probability for regularisation
    "learning_rate": 1e-3,           # Adam optimiser step size
    "batch_size": 1024,              # Larger batches exploit GPU parallelism
    "epochs": 50,                    # Maximum training epochs
    "patience": 10,                  # Early-stopping patience (epochs without
                                     # improvement on validation loss)
}


# ---------------------------------------------------------------------------
# 8.  EVALUATION THRESHOLDS
# ---------------------------------------------------------------------------

# Decision threshold for converting predicted probabilities to class labels.
# Default 0.5 is standard; tune toward lower values to increase recall
# (catch more attacks) at the cost of FPR (more false alarms).
DECISION_THRESHOLD: float = 0.5

# Number of cross-validation folds used during hyperparameter search
CV_FOLDS: int = 5


# ---------------------------------------------------------------------------
# 9.  LOGGING
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
