"""
=============================================================================
preprocessor.py — Leak-Proof Imputation, Scaling, and SMOTE
=============================================================================
Purpose:
    Transform raw feature matrices into model-ready arrays while STRICTLY
    preventing any information from the validation or test sets from
    influencing the transformation parameters.

Leak-Prevention Protocol:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  fit()   is called ONLY on X_train                                  │
    │  transform() is called on X_train, X_val, and X_test separately     │
    │  SMOTE is applied ONLY to the transformed (X_train, y_train)        │
    └─────────────────────────────────────────────────────────────────────┘

Pipeline components (in order):
    1. SimpleImputer   — fills NaN introduced by Inf-replacement
    2. StandardScaler  — zero-mean, unit-variance normalisation
    3. SMOTE           — over-samples minority class IN TRAIN ONLY

References:
    - Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 2011.
    - Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique",
      JAIR 2002.
    - He & Garcia, "Learning from Imbalanced Data", TKDE 2009.
=============================================================================
"""

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from pipeline.config import (
    ARTIFACT_DIR,
    GLOBAL_SEED,
    IMPUTATION_STRATEGY,
    SCALER,
    SMOTE_K_NEIGHBORS,
    USE_SMOTE,
)

logger = logging.getLogger(__name__)

# Path where the fitted preprocessor objects are persisted.
# During inference / deployment, load these rather than re-fitting.
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Stateful preprocessing pipeline that enforces the fit-on-train-only rule.

    Usage
    -----
        prep = Preprocessor()
        X_train_t, y_train_t = prep.fit_transform(X_train, y_train)
        X_val_t              = prep.transform(X_val)
        X_test_t             = prep.transform(X_test)   # No y needed

    After fitting, call prep.save() to persist for later inference.
    """

    def __init__(self) -> None:
        # Imputer: fills NaN with the column median (robust to outliers)
        self.imputer = SimpleImputer(
            strategy=IMPUTATION_STRATEGY,
            # Keep feature names so downstream diagnostics are readable
        )

        # Scaler: chosen via config
        if SCALER == "standard":
            self.scaler = StandardScaler()
        elif SCALER == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown SCALER='{SCALER}'. Choose 'standard' or 'minmax'."
            )

        # Stores the column names seen during fit (for validation)
        self._feature_names: list[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # fit_transform — call this ONLY on training data
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the imputer and scaler on X_train, then transform X_train.
        Optionally apply SMOTE to handle class imbalance.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (must NOT contain the label column).
        y_train : pd.Series
            Binary target vector aligned with X_train.

        Returns
        -------
        X_train_t : np.ndarray
            Preprocessed and (optionally) oversampled training features.
        y_train_t : np.ndarray
            Corresponding labels (length may be > original after SMOTE).
        """
        if self._fitted:
            raise RuntimeError(
                "Preprocessor.fit_transform() called more than once. "
                "Use transform() for validation and test data."
            )

        logger.info("Fitting preprocessor on training data ...")

        # Step 1 — Record feature names for reproducibility auditing
        self._feature_names = list(X_train.columns)

        # Step 2 — Imputation: fit on train, transform train
        #   The median of each column is computed exclusively from X_train.
        X_imputed = self.imputer.fit_transform(X_train)
        logger.info(
            "Imputation complete — strategy: %s | remaining NaN: %d",
            IMPUTATION_STRATEGY,
            np.isnan(X_imputed).sum(),
        )

        # Step 3 — Scaling: fit on train, transform train
        #   mean and std (or min/max) are computed exclusively from X_train.
        X_scaled = self.scaler.fit_transform(X_imputed)
        logger.info("Scaling complete — scaler: %s", SCALER)

        y_array = y_train.to_numpy()

        # Step 4 — SMOTE (optional, training only)
        if USE_SMOTE:
            X_scaled, y_array = _apply_smote(X_scaled, y_array)

        self._fitted = True
        logger.info(
            "Preprocessor fit complete. "
            "Final training set shape: X=%s | y=%s",
            X_scaled.shape, y_array.shape,
        )
        return X_scaled, y_array

    # ------------------------------------------------------------------
    # transform — call for validation and test data
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the ALREADY-FITTED imputer and scaler to X.
        No fitting occurs here — transformation parameters come from
        the training fit only.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must have the same columns as X_train).

        Returns
        -------
        X_t : np.ndarray
            Preprocessed features.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor.transform() called before fit_transform(). "
                "Call fit_transform(X_train, y_train) first."
            )

        # Guard: ensure column order matches training
        missing = set(self._feature_names) - set(X.columns)
        extra   = set(X.columns) - set(self._feature_names)
        if missing or extra:
            raise ValueError(
                f"Column mismatch!\n  Missing: {missing}\n  Extra: {extra}"
            )
        X = X[self._feature_names]   # enforce column order

        X_imputed = self.imputer.transform(X)
        X_scaled  = self.scaler.transform(X_imputed)
        return X_scaled

    # ------------------------------------------------------------------
    # Persistence helpers (for deployment)
    # ------------------------------------------------------------------

    def save(self, path: str = PREPROCESSOR_PATH) -> None:
        """
        Serialise the fitted preprocessor to disk with pickle.
        During inference / deployment, load this object rather than
        re-fitting from raw data.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved to: %s", path)

    @staticmethod
    def load(path: str = PREPROCESSOR_PATH) -> "Preprocessor":
        """Load a previously fitted Preprocessor from disk."""
        with open(path, "rb") as f:
            prep = pickle.load(f)
        if not prep._fitted:
            raise RuntimeError("Loaded preprocessor has not been fitted.")
        logger.info("Preprocessor loaded from: %s", path)
        return prep

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_names(self) -> list[str]:
        """Return the list of feature columns seen during training."""
        return self._feature_names

    def scaling_params(self) -> dict:
        """
        Return per-feature scaling parameters for reporting / auditing.
        Useful for journal appendices that require parameter disclosure.
        """
        if not self._fitted:
            return {}
        if isinstance(self.scaler, StandardScaler):
            return {
                "mean":  self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        elif isinstance(self.scaler, MinMaxScaler):
            return {
                "min":   self.scaler.data_min_.tolist(),
                "max":   self.scaler.data_max_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        return {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the training data to balance the class distribution.

    SMOTE generates synthetic samples of the minority class (ATTACK) by
    interpolating between existing minority instances in feature space.
    This avoids exact duplication (which would not add information) and
    is more principled than random over-sampling.

    IMPORTANT: SMOTE is applied here, AFTER imputation and scaling, so
    that distance calculations in the k-NN phase operate on comparable
    feature magnitudes.  Applying SMOTE before scaling would distort the
    synthetic samples.

    Parameters
    ----------
    X : np.ndarray  — Scaled training features.
    y : np.ndarray  — Corresponding binary labels.

    Returns
    -------
    X_res, y_res : np.ndarray — Resampled arrays with balanced classes.
    """
    before = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(
        "Applying SMOTE — class distribution before: %s", before
    )

    smote = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X, y)

    after = dict(zip(*np.unique(y_res, return_counts=True)))
    logger.info(
        "SMOTE complete — class distribution after:  %s | "
        "new samples generated: %d",
        after,
        len(y_res) - len(y),
    )
    return X_res, y_res
