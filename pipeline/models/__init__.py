"""
models/ — Individual model implementations for the IDS pipeline.

Each module exposes a single public class that:
  - Accepts numpy arrays (X_train, y_train, X_val, y_val)
  - Exposes fit(), predict(), predict_proba(), and save() methods
  - Logs validation metrics during training for early-stopping decisions
"""

from pipeline.models.random_forest  import RandomForestModel
from pipeline.models.xgboost_model  import XGBoostModel
from pipeline.models.deep_learning  import DeepLearningModel

__all__ = ["RandomForestModel", "XGBoostModel", "DeepLearningModel"]
