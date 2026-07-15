"""
=============================================================================
main.py — End-to-End Pipeline Entry Point
=============================================================================
AI-Driven Network Traffic Anomaly Detection
University Cybersecurity Curriculum — Academic Training Pipeline

Usage (local):
    python main.py

Usage (Docker):
    docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts \
        ids-pipeline python main.py

Usage (AWS SageMaker Script Mode):
    estimator = SKLearn(entry_point="main.py", ...)
    estimator.fit({"train": s3_data_uri})

Pipeline stages (in order):
    1. Load & clean CICIDS2017 CSVs
    2. Binarise labels (BENIGN=0, ATTACK=1)
    3. Chronological OR stratified train / val / test split
       *** Test set is ISOLATED here — never touched again until evaluation ***
    4. Fit preprocessor on train only (impute + scale)
    5. Apply SMOTE to training set only
    6. Train Random Forest, XGBoost, Deep Learning models
    7. Evaluate each model on the ISOLATED test set
    8. Generate comparison table, plots, and JSON reports

Author note:
    Every parameter used in this pipeline is defined in pipeline/config.py
    and documented in-line.  Figures and JSON reports are written to
    artifacts/reports/ and artifacts/plots/.
=============================================================================
"""

import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Logging setup — must be configured before any pipeline imports so that
# all module-level loggers inherit the handler.
# ---------------------------------------------------------------------------
from pipeline.config import ARTIFACT_DIR, LOG_LEVEL

# Create the artifacts directory BEFORE opening the log file handler
os.makedirs(ARTIFACT_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Also write to a persistent log file inside the artifact directory
        logging.FileHandler(
            os.path.join(ARTIFACT_DIR, "pipeline_run.log"), mode="w"
        ),
    ],
)

logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Pipeline imports (after logging config)
# ---------------------------------------------------------------------------
from pipeline.data_loader  import load_and_split
from pipeline.preprocessor import Preprocessor
from pipeline.trainer      import train_all
from pipeline.evaluator    import evaluate, compare_models
from pipeline.config       import DECISION_THRESHOLD


# ---------------------------------------------------------------------------
# Ensure all output directories exist
# ---------------------------------------------------------------------------
os.makedirs(os.path.join(ARTIFACT_DIR, "models"),  exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "reports"), exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "plots"),   exist_ok=True)


def main() -> None:
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("  AI-Driven Network Traffic Anomaly Detection — Training Pipeline")
    logger.info("=" * 70)

    # =========================================================================
    # STAGE 1 & 2: Load, clean, and binarise labels
    #              Chronological / stratified train–val–test split
    #              *** TEST SET IS SEALED AFTER THIS CALL ***
    # =========================================================================
    logger.info("\n[STAGE 1/4] Data loading and splitting ...")
    t0 = time.time()

    X_train_raw, X_val_raw, X_test_raw, \
    y_train, y_val, y_test = load_and_split()

    logger.info("Stage 1 complete in %.1f s.\n", time.time() - t0)

    # =========================================================================
    # STAGE 3: Preprocessing — fit on train ONLY, transform all three splits
    # =========================================================================
    logger.info("[STAGE 2/4] Preprocessing (imputation → scaling → SMOTE) ...")
    t0 = time.time()

    prep = Preprocessor()

    # fit_transform: fits the imputer and scaler on X_train, applies SMOTE
    X_train, y_train_bal = prep.fit_transform(X_train_raw, y_train)

    # transform: applies the ALREADY-FITTED transformers — no new parameters
    X_val  = prep.transform(X_val_raw)
    X_test = prep.transform(X_test_raw)    # ← test set transformed, NOT fitted

    # Persist the preprocessor for later inference / deployment
    prep.save()

    logger.info("Stage 2 complete in %.1f s.\n", time.time() - t0)

    # =========================================================================
    # STAGE 4: Train all three models
    # =========================================================================
    logger.info("[STAGE 3/4] Model training ...")
    t0 = time.time()

    rf_model, xgb_model, dl_model = train_all(
        X_train, y_train_bal,   # balanced (SMOTE) training data
        X_val,   y_val.to_numpy(),   # unbalanced validation (real distribution)
    )

    logger.info("Stage 3 complete in %.1f s.\n", time.time() - t0)

    # =========================================================================
    # STAGE 5: Evaluation on the ISOLATED test set
    #          This is the only time the test set is used — ever.
    # =========================================================================
    logger.info("[STAGE 4/4] Evaluating all models on ISOLATED test set ...")
    logger.info(
        "Test set size: %d samples | Attack rate: %.2f%%",
        len(y_test), y_test.mean() * 100,
    )

    y_test_np = y_test.to_numpy()
    feature_names = prep.feature_names()

    # Random Forest evaluation
    rf_proba  = rf_model.predict_proba(X_test)
    rf_report = evaluate(
        model_name="RandomForest",
        y_true=y_test_np,
        y_proba=rf_proba,
        threshold=DECISION_THRESHOLD,
        feature_names=feature_names,
    )

    # XGBoost evaluation
    xgb_proba  = xgb_model.predict_proba(X_test)
    xgb_report = evaluate(
        model_name="XGBoost",
        y_true=y_test_np,
        y_proba=xgb_proba,
        threshold=DECISION_THRESHOLD,
        feature_names=feature_names,
    )

    # Deep Learning evaluation
    dl_proba  = dl_model.predict_proba(X_test)
    dl_report = evaluate(
        model_name="DeepLearning",
        y_true=y_test_np,
        y_proba=dl_proba,
        threshold=DECISION_THRESHOLD,
        feature_names=feature_names,
    )

    # =========================================================================
    # Side-by-side comparison table for journal / paper results section
    # =========================================================================
    compare_models([rf_report, xgb_report, dl_report])

    # =========================================================================
    # Pipeline summary
    # =========================================================================
    total_time = time.time() - pipeline_start
    logger.info("\n%s", "=" * 70)
    logger.info("  Pipeline completed successfully in %.1f s (%.1f min).",
                total_time, total_time / 60)
    logger.info("  Artefacts written to: %s", ARTIFACT_DIR)
    logger.info("    ├── models/        — trained model files")
    logger.info("    ├── reports/       — JSON evaluation reports + comparison")
    logger.info("    ├── plots/         — ROC, PR, and confusion matrix figures")
    logger.info("    └── pipeline_run.log")
    logger.info("%s\n", "=" * 70)


if __name__ == "__main__":
    main()
