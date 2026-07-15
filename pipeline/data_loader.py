"""
=============================================================================
data_loader.py — Dataset Ingestion, Cleaning, and Leak-Proof Splitting
=============================================================================
Purpose:
    1. Load all CICIDS2017 day-CSVs into a single Pandas DataFrame.
    2. Perform lightweight cleaning (infinite values, duplicate rows).
    3. Encode the multi-class attack label into a binary target
       (0 = BENIGN, 1 = ATTACK) suitable for anomaly detection.
    4. Produce a STRICTLY ISOLATED test set BEFORE any preprocessing
       touches the data.  This is the single most important step for
       preventing data leakage in an IDS evaluation.

Data Leakage Prevention Principle:
    "No information about the test set may flow into the training
    pipeline."  Concretely:
      - The StandardScaler is fit ONLY on X_train (see preprocessor.py).
      - SMOTE is applied ONLY to (X_train, y_train).
      - The test set is sealed here and not unwrapped until evaluation.

Dataset:
    CICIDS2017 — Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A.A.
    "Toward Generating a New Intrusion Detection Dataset and Intrusion
    Traffic Characterization." ICISSP 2018.
    URL: https://www.unb.ca/cic/datasets/ids-2017.html

    79 network-flow features extracted by CICFlowMeter covering:
      - Flow duration, packet length statistics
      - Inter-arrival times (IAT) forward/backward
      - TCP flag counts, window sizes
      - Active / idle time distributions
=============================================================================
"""

import glob
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import (
    BENIGN_LABEL,
    CHRONOLOGICAL_SPLIT,
    COLS_TO_DROP,
    DATASET_GLOB,
    GLOBAL_SEED,
    LABEL_COL,
    RAW_LABEL_COL,
    TEST_SIZE,
    TIMESTAMP_COL,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_split() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,   # X_train, X_val, X_test
    pd.Series,   pd.Series,   pd.Series,        # y_train, y_val, y_test
]:
    """
    Master entry point for data loading.

    Returns
    -------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices for training, validation, and test sets.
        The test set is completely isolated from any fit operations.
    y_train, y_val, y_test : pd.Series
        Binary target vectors (0 = BENIGN, 1 = ATTACK).
    """
    df = _load_raw()
    df = _clean(df)
    df = _binarise_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test = _split(df)

    # Report class distribution so analysts can spot severe imbalance
    _log_split_stats(y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Step 1 — Raw data ingestion
# ---------------------------------------------------------------------------

def _load_raw() -> pd.DataFrame:
    """
    Glob all CICIDS2017 day-CSV files and concatenate into one DataFrame.

    The dataset is distributed as separate files per capture day:
        Monday-WorkingHours.pcap_ISCX.csv    (Benign only)
        Tuesday-WorkingHours.pcap_ISCX.csv   (Benign + FTP/SSH Brute Force)
        Wednesday-...                         (Benign + DoS / Slowloris)
        Thursday-...                          (Benign + Web Attacks)
        Friday-...                            (Benign + DDoS / PortScan / Botnet)

    Each file has 79 feature columns + 1 label column. Column names carry
    a leading/trailing space in the original CICFlowMeter output — we strip
    these after loading.
    """
    csv_files = sorted(glob.glob(DATASET_GLOB))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found matching: {DATASET_GLOB}\n"
            "Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html "
            "and place the CSVs in the directory pointed to by DATA_DIR "
            "(default: data/raw/)."
        )

    logger.info("Found %d CSV file(s) to load.", len(csv_files))
    frames = []

    for path in csv_files:
        logger.info("  Loading: %s", os.path.basename(path))
        # low_memory=False avoids mixed-type inference warnings in large files
        chunk = pd.read_csv(path, low_memory=False)
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Raw dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Cleaning
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight, label-agnostic cleaning steps.

    Operations performed:
    1. Strip whitespace from column names (CICFlowMeter artifact).
    2. Replace +Inf / -Inf with NaN so imputers handle them uniformly.
    3. Drop exact duplicate rows (CICFlowMeter can emit duplicates when
       bidirectional flows are captured on both NICs).
    4. Drop columns that are leakage-prone or non-informative per config.
    """
    # --- 2a. Strip leading/trailing whitespace from column names ----------
    df.columns = df.columns.str.strip()

    # After stripping, the label column name changes from " Label" to "Label"
    # Normalise label column name used in the rest of the pipeline
    if "Label" in df.columns and LABEL_COL not in df.columns:
        df.rename(columns={"Label": LABEL_COL}, inplace=True)
    elif RAW_LABEL_COL.strip() in df.columns:
        df.rename(columns={RAW_LABEL_COL.strip(): LABEL_COL}, inplace=True)

    # Similarly normalise the timestamp column name after stripping
    ts_col_stripped = TIMESTAMP_COL.strip()
    if ts_col_stripped in df.columns and TIMESTAMP_COL not in df.columns:
        df.rename(columns={ts_col_stripped: TIMESTAMP_COL}, inplace=True)

    # --- 2b. Replace infinities with NaN ----------------------------------
    # CICIDS2017 contains Flow Bytes/s = Inf when flow duration == 0 ms.
    # These are not truly missing — they represent extremely short flows —
    # but they cannot be fed to most ML algorithms as-is.
    n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if n_inf > 0:
        logger.warning("Replacing %d infinite values with NaN.", n_inf)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 2c. Drop duplicate rows ------------------------------------------
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.info("Dropped %d duplicate rows.", n_dupes)

    # --- 2d. Drop leakage-prone / non-informative columns ----------------
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_present, inplace=True, errors="ignore")
    logger.info("Dropped columns: %s", cols_present)

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 3 — Label binarisation
# ---------------------------------------------------------------------------

def _binarise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the multi-class attack label into a binary target.

    CICIDS2017 label values (examples):
        BENIGN, DDoS, PortScan, FTP-Patator, SSH-Patator,
        DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS GoldenEye,
        Heartbleed, Web Attack – Brute Force, Web Attack – XSS,
        Web Attack – Sql Injection, Infiltration, Bot

    For anomaly detection we treat ALL non-benign traffic as class 1.
    A multi-class extension is straightforward: replace the binary
    encoding below with LabelEncoder() on the raw label.

    Security interpretation:
        Class 0 (BENIGN)  → Normal traffic — should NOT trigger an alert.
        Class 1 (ATTACK)  → Malicious traffic — must be flagged.
    """
    label_counts = df[LABEL_COL].value_counts()
    logger.info("Raw label distribution:\n%s", label_counts.to_string())

    # Binary encoding: BENIGN → 0, everything else → 1
    df[LABEL_COL] = (df[LABEL_COL] != BENIGN_LABEL).astype(int)

    attack_rate = df[LABEL_COL].mean() * 100
    logger.info(
        "Binary label distribution — BENIGN: %.1f%%, ATTACK: %.1f%%",
        100 - attack_rate,
        attack_rate,
    )
    return df


# ---------------------------------------------------------------------------
# Step 4 — Leak-proof train / validation / test split
# ---------------------------------------------------------------------------

def _split(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,   pd.Series,   pd.Series,
]:
    """
    Produce three non-overlapping, leak-proof splits.

    Strategy A — Chronological (preferred when CHRONOLOGICAL_SPLIT=True):
        Sort the full dataset by timestamp before splitting.  The last
        TEST_SIZE fraction becomes the test set; the preceding VAL_SIZE
        fraction becomes validation; the remainder is training.
        This simulates a zero-day scenario where the model is evaluated
        on network traffic it has never been exposed to in time.
        No shuffle is applied, preserving temporal order.

    Strategy B — Stratified random (fallback):
        Use sklearn's train_test_split with stratification to maintain
        the original class ratio in all three splits.  A fixed random
        seed guarantees reproducibility.

    WHY SPLIT BEFORE PREPROCESSING?
        Any scaler, imputer, or SMOTE fitted on the full dataset would
        encode distributional information from the test set into the
        parameters used to transform the training set.  This inflates
        performance metrics and violates the independence assumption.
        Test set isolation here prevents that entirely.
    """
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL]

    if CHRONOLOGICAL_SPLIT and TIMESTAMP_COL in df.columns:
        # ----------------------------------------------------------------
        # Strategy A: Chronological split
        # ----------------------------------------------------------------
        logger.info(
            "Using CHRONOLOGICAL split (TIMESTAMP_COL='%s').", TIMESTAMP_COL
        )
        # Sort ascending — oldest traffic first
        sort_idx = df[TIMESTAMP_COL].argsort()
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y.iloc[sort_idx].reset_index(drop=True)

        n = len(y)
        test_start  = int(n * (1 - TEST_SIZE))
        val_start   = int(n * (1 - TEST_SIZE - VAL_SIZE))

        X_test,  y_test  = X.iloc[test_start:],        y.iloc[test_start:]
        X_val,   y_val   = X.iloc[val_start:test_start], y.iloc[val_start:test_start]
        X_train, y_train = X.iloc[:val_start],          y.iloc[:val_start]

    else:
        # ----------------------------------------------------------------
        # Strategy B: Stratified random split
        # ----------------------------------------------------------------
        logger.info("Using STRATIFIED RANDOM split.")

        # First carve out the test set (isolated)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=GLOBAL_SEED,
        )

        # From the remaining data, carve out the validation set
        # The relative size of val within the temp set:
        #   val_relative = VAL_SIZE / (1 - TEST_SIZE)
        val_relative = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            stratify=y_temp,
            random_state=GLOBAL_SEED,
        )

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _log_split_stats(
    y_train: pd.Series,
    y_val:   pd.Series,
    y_test:  pd.Series,
) -> None:
    """Log per-split class distributions for audit purposes."""
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = y.value_counts().sort_index()
        total  = len(y)
        logger.info(
            "%s set — BENIGN: %d (%.1f%%), ATTACK: %d (%.1f%%)",
            name,
            counts.get(0, 0), counts.get(0, 0) / total * 100,
            counts.get(1, 0), counts.get(1, 0) / total * 100,
        )
