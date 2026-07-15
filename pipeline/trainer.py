"""
=============================================================================
trainer.py — Training Orchestration Across All Models
=============================================================================
Purpose:
    Coordinates the training of all three models (Random Forest, XGBoost,
    Deep Learning) and returns their evaluation-ready outputs.

    This module sits between the preprocessor (which produces arrays) and
    the evaluator (which consumes predictions).  It does NOT touch the test
    set — that responsibility belongs entirely to evaluator.py.

Design:
    - Models are trained sequentially on the same preprocessed training data.
    - Each model receives the validation set for early stopping / diagnostics
      only.  The validation set is never used for hyperparameter selection
      in this script (that would require a nested CV wrapper).
    - All trained models are persisted to ARTIFACT_DIR/models/.
=============================================================================
"""

import logging
import os
from typing import Tuple

import numpy as np

from pipeline.config import ARTIFACT_DIR
from pipeline.models import DeepLearningModel, RandomForestModel, XGBoostModel

logger = logging.getLogger(__name__)


def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> Tuple[RandomForestModel, XGBoostModel, DeepLearningModel]:
    """
    Train all three models and return fitted instances.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Preprocessed (and SMOTE-balanced) training arrays.
    X_val, y_val : np.ndarray
        Preprocessed validation arrays — NOT balanced; reflects real
        class distribution for unbiased early-stopping signals.

    Returns
    -------
    rf_model  : Fitted RandomForestModel
    xgb_model : Fitted XGBoostModel
    dl_model  : Fitted DeepLearningModel
    """
    os.makedirs(os.path.join(ARTIFACT_DIR, "models"), exist_ok=True)

    # ------------------------------------------------------------------
    # Model 1 — Random Forest
    #   Fastest to train; provides feature importance baseline.
    # ------------------------------------------------------------------
    logger.info("\n%s\n  Training Model 1/3: Random Forest\n%s", "=" * 60, "=" * 60)
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train, X_val, y_val)
    rf_model.save()

    # ------------------------------------------------------------------
    # Model 2 — XGBoost
    #   Gradient boosting with early stopping on validation logloss.
    # ------------------------------------------------------------------
    logger.info("\n%s\n  Training Model 2/3: XGBoost\n%s", "=" * 60, "=" * 60)
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_model.save()

    # ------------------------------------------------------------------
    # Model 3 — Deep Learning (TensorFlow MLP)
    #   Most expressive; catches subtle attack patterns.
    # ------------------------------------------------------------------
    logger.info("\n%s\n  Training Model 3/3: Deep Learning (MLP)\n%s", "=" * 60, "=" * 60)
    dl_model = DeepLearningModel()
    dl_model.fit(X_train, y_train, X_val, y_val)
    dl_model.save()

    logger.info("\nAll models trained and saved to: %s/models/", ARTIFACT_DIR)
    return rf_model, xgb_model, dl_model
