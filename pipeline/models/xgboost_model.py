"""
=============================================================================
xgboost_model.py — XGBoost Classifier for Network Anomaly Detection
=============================================================================
Purpose:
    Implements the XGBoost model — consistently one of the top performers
    on tabular network intrusion datasets including CICIDS2017 and UNSW-NB15.

    Key advantages for IDS:
      - Gradient boosting sequentially corrects residual errors, yielding
        very low false-negative rates (missed attacks).
      - Built-in L1/L2 regularisation (reg_alpha, reg_lambda) mitigates
        overfitting on high-dimensional flow features.
      - Native support for early stopping on a held-out validation set
        avoids training too many boosting rounds.
      - Tree SHAP values (Lundberg & Lee, 2017) can explain individual
        predictions — a requirement for responsible IDS deployment.

References:
    Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
    System. KDD 2016. https://doi.org/10.1145/2939672.2939785

    Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
    Interpreting Model Predictions. NeurIPS 2017.
=============================================================================
"""

import logging
import os

import numpy as np
from xgboost import XGBClassifier

from pipeline.config import (
    ARTIFACT_DIR,
    GLOBAL_SEED,
    XGB_EARLY_STOPPING_ROUNDS,
    XGB_PARAMS,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(ARTIFACT_DIR, "models", "xgboost.json")


class XGBoostModel:
    """
    XGBoost binary classifier with early stopping on the validation set.
    """

    def __init__(self) -> None:
        # XGBClassifier wraps the C++ booster with a sklearn-compatible API.
        # We pass all hyperparameters explicitly for reproducibility.
        self.clf = XGBClassifier(**XGB_PARAMS)
        self.name = "XGBoost"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> "XGBoostModel":
        """
        Fit XGBoost with early stopping monitored on the validation set.

        Early stopping:
            If the validation loss (logloss) does not improve for
            XGB_EARLY_STOPPING_ROUNDS consecutive boosting rounds, training
            halts and the model reverts to the best checkpoint.  This:
              1. Prevents overfitting to training noise.
              2. Reduces unnecessary computation on EC2/SageMaker.

        NOTE: The validation set here is used ONLY for early stopping — NOT
        for any hyperparameter tuning.  The test set remains fully isolated.
        """
        logger.info(
            "[%s] Starting training | n_estimators=%d | max_depth=%d | "
            "learning_rate=%.4f | n_samples=%d",
            self.name,
            XGB_PARAMS["n_estimators"],
            XGB_PARAMS["max_depth"],
            XGB_PARAMS["learning_rate"],
            X_train.shape[0],
        )

        self.clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            # early_stopping_rounds: patience counter reset if eval metric improves
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            verbose=False,   # suppress per-round output; our logger handles this
        )

        best_iter = self.clf.best_iteration
        best_score = self.clf.best_score
        logger.info(
            "[%s] Training complete — best_iteration: %d | best_val_logloss: %.4f",
            self.name, best_iter, best_score,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard class labels (0 or 1), using the best booster checkpoint."""
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of ATTACK class — shape (n_samples,)."""
        return self.clf.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = MODEL_PATH) -> None:
        """
        Save the booster in XGBoost's native JSON format.
        JSON is preferred over pickle for XGBoost because:
          - It is version-agnostic (pickle format can break across XGBoost versions).
          - It is human-readable, which aids auditing.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.clf.save_model(path)
        logger.info("[%s] Model saved to: %s", self.name, path)

    def load(self, path: str = MODEL_PATH) -> "XGBoostModel":
        """Load a booster from XGBoost JSON format."""
        self.clf.load_model(path)
        logger.info("[%s] Model loaded from: %s", self.name, path)
        return self
