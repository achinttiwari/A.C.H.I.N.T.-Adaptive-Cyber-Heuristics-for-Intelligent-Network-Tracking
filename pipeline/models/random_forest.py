"""
=============================================================================
random_forest.py — Random Forest Classifier for Network Anomaly Detection
=============================================================================
Purpose:
    Implements the Random Forest baseline model.  Random Forests are a
    natural first choice for IDS tasks because:
      - They are robust to noisy features (irrelevant features are
        rarely selected at every split).
      - They provide a built-in feature importance ranking, which is
        valuable for explainability in academic and operational settings.
      - They handle class imbalance via the class_weight parameter.
      - Training is embarrassingly parallel (n_jobs=-1 uses all cores).

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    https://doi.org/10.1023/A:1010933404324

    For IDS application see:
    Yin, C. et al. (2017). "A Deep Learning Approach for Intrusion
    Detection Using Recurrent Neural Networks." IEEE Access, 5, 21954–21961.
=============================================================================
"""

import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
)

from pipeline.config import ARTIFACT_DIR, RF_PARAMS

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "models", "random_forest.pkl")


class RandomForestModel:
    """
    Thin wrapper around sklearn's RandomForestClassifier that adds
    logging, persistence, and a consistent API shared by all model classes
    in this pipeline.
    """

    def __init__(self) -> None:
        # Instantiate classifier with all hyperparameters from config.py.
        # Every parameter is deliberately explicit — no hidden defaults —
        # to satisfy journal reproducibility requirements.
        self.clf = RandomForestClassifier(**RF_PARAMS)
        self.name = "RandomForest"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "RandomForestModel":
        """
        Fit the Random Forest on (X_train, y_train) and log held-out
        validation metrics.  The validation set is NOT used to tune
        hyperparameters here — it is used only for early diagnostic
        output.  Hyperparameter search (if desired) should be wrapped
        around this class using sklearn's GridSearchCV / RandomizedSearchCV.

        Parameters
        ----------
        X_train, y_train : np.ndarray  — Training data (post-preprocessing).
        X_val,   y_val   : np.ndarray  — Validation data (not seen by scaler
                                          or SMOTE during preprocessing).
        Returns
        -------
        self : RandomForestModel (for method chaining)
        """
        logger.info(
            "[%s] Starting training | n_estimators=%d | n_samples=%d | "
            "n_features=%d",
            self.name, RF_PARAMS["n_estimators"], X_train.shape[0], X_train.shape[1],
        )

        self.clf.fit(X_train, y_train)
        logger.info("[%s] Training complete.", self.name)

        # Evaluate on validation set for early sanity check
        y_val_pred = self.clf.predict(X_val)
        y_val_prob = self.clf.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_prob)
        logger.info(
            "[%s] Validation — Accuracy: %.4f | AUC-ROC: %.4f", self.name, acc, auc
        )

        # Log top-10 most important features (aids interpretability)
        importances = self.clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        logger.info(
            "[%s] Top-10 feature importance indices (descending): %s",
            self.name, top_indices.tolist(),
        )

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions (0 or 1)."""
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities — shape (n_samples, 2).
        Column 1 is the probability of ATTACK, used for ROC-AUC and
        threshold sweeps.
        """
        return self.clf.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Feature importances (for academic reporting)
    # ------------------------------------------------------------------

    def feature_importances(self) -> np.ndarray:
        """
        Gini-impurity-based feature importance scores.
        Normalised so they sum to 1.0.
        """
        return self.clf.feature_importances_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_PATH) -> None:
        """Serialise the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[%s] Model saved to: %s", self.name, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "RandomForestModel":
        """Load a fitted RandomForestModel from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("[RandomForest] Model loaded from: %s", path)
        return model
