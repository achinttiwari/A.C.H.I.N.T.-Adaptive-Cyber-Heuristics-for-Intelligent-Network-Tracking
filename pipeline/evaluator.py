"""
=============================================================================
evaluator.py — Comprehensive Test-Set Evaluation and Report Generation
=============================================================================
Purpose:
    Compute and persist every evaluation metric required for a high-impact
    cybersecurity / AI journal submission.  All metrics are computed
    EXCLUSIVELY on the held-out test set — the test set is NEVER used
    during training, validation, or hyperparameter selection.

Metrics computed:
    1.  Confusion Matrix (raw counts and normalised)
    2.  Accuracy
    3.  Precision, Recall, F1-Score (macro, micro, weighted, per-class)
    4.  False Positive Rate (FPR) — critical for operational IDS
    5.  False Negative Rate (FNR) — missed attacks
    6.  Specificity (True Negative Rate)
    7.  Matthews Correlation Coefficient (MCC) — balanced for imbalanced data
    8.  Cohen's Kappa
    9.  ROC Curve + AUC-ROC
    10. Precision–Recall Curve + AP (Average Precision)

Output artefacts:
    - JSON report  → artifacts/reports/<model_name>_report.json
    - ROC curve    → artifacts/plots/<model_name>_roc.png
    - PR curve     → artifacts/plots/<model_name>_pr_curve.png
    - Confusion matrix heatmap → artifacts/plots/<model_name>_cm.png

Reference:
    Davis, J., & Goadrich, M. (2006). The Relationship Between
    Precision-Recall and ROC Curves. ICML 2006.

    Chicco, D., & Jurman, G. (2020). The advantages of the Matthews
    correlation coefficient (MCC) over F1 score and accuracy.
    BMC Genomics, 21(1), 6.
=============================================================================
"""

import json
import logging
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for server / cloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from pipeline.config import DECISION_THRESHOLD, PLOT_DIR, REPORT_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    model_name:  str,
    y_true:      np.ndarray,
    y_proba:     np.ndarray,
    threshold:   float = DECISION_THRESHOLD,
    feature_names: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Full evaluation suite on the test set.

    Parameters
    ----------
    model_name    : Human-readable name used in filenames and reports.
    y_true        : Ground-truth binary labels (0=BENIGN, 1=ATTACK).
    y_proba       : Predicted probability of ATTACK class — shape (n,).
    threshold     : Decision threshold for converting probabilities to labels.
    feature_names : Optional list for axis labels (not used in metrics).

    Returns
    -------
    report : dict
        All computed metrics in a serialisable dictionary.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,   exist_ok=True)

    # Convert probabilities to hard labels at the chosen threshold
    y_pred = (y_proba >= threshold).astype(int)

    logger.info("=" * 70)
    logger.info("EVALUATION REPORT — %s (threshold=%.2f)", model_name, threshold)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Confusion Matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(
        "\nConfusion Matrix:\n"
        "                Predicted BENIGN  Predicted ATTACK\n"
        "  True BENIGN        %8d        %8d\n"
        "  True ATTACK        %8d        %8d",
        tn, fp, fn, tp
    )

    # ------------------------------------------------------------------
    # 2. Core metrics derived from the confusion matrix
    # ------------------------------------------------------------------
    accuracy    = accuracy_score(y_true, y_pred)
    # Precision: of all predicted attacks, how many are real?
    precision   = tp / (tp + fp + 1e-10)
    # Recall / True Positive Rate / Sensitivity: of all real attacks, how many
    # did we catch?  A low recall means many attacks go undetected — dangerous.
    recall_tpr  = tp / (tp + fn + 1e-10)
    # F1: harmonic mean of precision and recall
    f1          = 2 * precision * recall_tpr / (precision + recall_tpr + 1e-10)
    # False Positive Rate: of all benign flows, how many were wrongly flagged?
    # A high FPR means too many false alarms — operators stop trusting the system.
    fpr_val     = fp / (fp + tn + 1e-10)
    # False Negative Rate: of all attacks, how many were missed?
    fnr_val     = fn / (fn + tp + 1e-10)
    # Specificity (True Negative Rate): ability to correctly identify benign.
    specificity = tn / (tn + fp + 1e-10)
    # Matthews Correlation Coefficient: balanced metric even for imbalanced data.
    mcc         = matthews_corrcoef(y_true, y_pred)
    # Cohen's Kappa: agreement beyond chance.
    kappa       = cohen_kappa_score(y_true, y_pred)

    logger.info(
        "\nCore Metrics:\n"
        "  Accuracy         : %.4f\n"
        "  Precision        : %.4f\n"
        "  Recall (TPR)     : %.4f\n"
        "  F1-Score         : %.4f\n"
        "  False Positive Rate (FPR)  : %.4f  ← CRITICAL for IDS\n"
        "  False Negative Rate (FNR)  : %.4f\n"
        "  Specificity (TNR): %.4f\n"
        "  MCC              : %.4f\n"
        "  Cohen's Kappa    : %.4f",
        accuracy, precision, recall_tpr, f1,
        fpr_val, fnr_val, specificity, mcc, kappa,
    )

    # ------------------------------------------------------------------
    # 3. sklearn's full classification report (per-class + macro/weighted)
    # ------------------------------------------------------------------
    clf_report = classification_report(
        y_true, y_pred,
        target_names=["BENIGN", "ATTACK"],
        digits=4,
    )
    logger.info("\nFull Classification Report:\n%s", clf_report)

    # ------------------------------------------------------------------
    # 4. AUC-ROC
    # ------------------------------------------------------------------
    auc_roc = roc_auc_score(y_true, y_proba)
    logger.info("AUC-ROC: %.4f", auc_roc)

    # ------------------------------------------------------------------
    # 5. Average Precision (AUC-PR)
    # ------------------------------------------------------------------
    auc_pr = average_precision_score(y_true, y_proba)
    logger.info("AUC-PR (Average Precision): %.4f", auc_pr)

    # ------------------------------------------------------------------
    # 6. Assemble report dictionary
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "model_name"        : model_name,
        "threshold"         : threshold,
        "n_test_samples"    : int(len(y_true)),
        "n_attack_samples"  : int(y_true.sum()),
        "n_benign_samples"  : int((y_true == 0).sum()),
        "confusion_matrix"  : {"tn": int(tn), "fp": int(fp),
                                "fn": int(fn), "tp": int(tp)},
        "accuracy"          : round(float(accuracy),    4),
        "precision"         : round(float(precision),   4),
        "recall_tpr"        : round(float(recall_tpr),  4),
        "f1_score"          : round(float(f1),          4),
        "false_positive_rate": round(float(fpr_val),    4),
        "false_negative_rate": round(float(fnr_val),    4),
        "specificity_tnr"   : round(float(specificity), 4),
        "mcc"               : round(float(mcc),         4),
        "cohen_kappa"       : round(float(kappa),       4),
        "auc_roc"           : round(float(auc_roc),     4),
        "auc_pr"            : round(float(auc_pr),      4),
        "classification_report": clf_report,
    }

    # Persist JSON
    json_path = os.path.join(REPORT_DIR, f"{model_name}_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to: %s", json_path)

    # ------------------------------------------------------------------
    # 7. Generate and save plots
    # ------------------------------------------------------------------
    _plot_confusion_matrix(cm, model_name)
    _plot_roc_curve(y_true, y_proba, auc_roc, model_name)
    _plot_pr_curve(y_true, y_proba, auc_pr, model_name)

    return report


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> None:
    """
    Save a colour-coded confusion matrix heatmap.

    The matrix shows:
      Row 0: True BENIGN  → [TN, FP]
      Row 1: True ATTACK  → [FN, TP]

    Colour interpretation:
      - High TN and TP (diagonal) are desirable — darker green.
      - High FP: many false alarms — highlighted in red.
      - High FN: many missed attacks — the worst outcome for a security system.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ["BENIGN", "ATTACK"]

    for ax, (title, data, fmt) in zip(
        axes,
        [
            ("Confusion Matrix (Counts)",       cm,                    "d"),
            ("Confusion Matrix (Normalised)",   cm / cm.sum(axis=1, keepdims=True), ".2%"),
        ],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{model_name} — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(2):
            for j in range(2):
                text = format(data[i, j], fmt)
                colour = "white" if data[i, j] > data.max() / 2 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        color=colour, fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_cm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix plot saved to: %s", path)


def _plot_roc_curve(
    y_true:     np.ndarray,
    y_proba:    np.ndarray,
    auc_roc:    float,
    model_name: str,
) -> None:
    """
    ROC Curve plot (FPR on x-axis, TPR on y-axis).

    The ROC curve shows the trade-off between the True Positive Rate
    (sensitivity) and the False Positive Rate at every possible decision
    threshold.  The area under this curve (AUC-ROC) summarises overall
    discrimination ability:
      - AUC = 1.0 → perfect classifier
      - AUC = 0.5 → random classifier (diagonal line)

    For IDS evaluation, the region of the curve where FPR < 0.01 is
    especially important — operational systems require a very low FPR.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="#1a6fb5",
            label=f"ROC (AUC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    # Shade the area under the curve
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.1, color="#1a6fb5")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
    ax.set_title(f"{model_name} — ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to: %s", path)


def _plot_pr_curve(
    y_true:     np.ndarray,
    y_proba:    np.ndarray,
    auc_pr:     float,
    model_name: str,
) -> None:
    """
    Precision–Recall Curve plot.

    The PR curve is more informative than ROC when classes are imbalanced
    (which is the typical case in network IDS — attacks are rare relative
    to benign traffic).  A large area under the PR curve indicates the
    model maintains high precision even at high recall levels.

    Reference:
        Davis, J., & Goadrich, M. (2006). The Relationship Between
        Precision-Recall and ROC Curves. ICML 2006.
    """
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)

    # Baseline: fraction of positives in the dataset
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_arr, precision_arr, lw=2, color="#d62728",
            label=f"PR (AP = {auc_pr:.4f})")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Baseline (attack rate = {baseline:.3f})")

    ax.fill_between(recall_arr, precision_arr, alpha=0.1, color="#d62728")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=12)
    ax.set_ylabel("Precision",                   fontsize=12)
    ax.set_title(f"{model_name} — Precision–Recall Curve",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{model_name}_pr_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved to: %s", path)


# ---------------------------------------------------------------------------
# Multi-model comparison utility
# ---------------------------------------------------------------------------

def compare_models(reports: list[Dict[str, Any]]) -> None:
    """
    Print a side-by-side comparison table for all evaluated models.
    Useful for the Results section of a paper.
    """
    metrics = [
        "accuracy", "precision", "recall_tpr", "f1_score",
        "false_positive_rate", "false_negative_rate",
        "specificity_tnr", "mcc", "auc_roc", "auc_pr",
    ]

    header = f"{'Metric':<28}" + "".join(
        f"{r['model_name']:>16}" for r in reports
    )
    separator = "-" * len(header)

    lines = ["\n" + separator, "MODEL COMPARISON TABLE", separator, header, separator]

    for m in metrics:
        row = f"{m:<28}" + "".join(
            f"{r.get(m, float('nan')):>16.4f}" for r in reports
        )
        lines.append(row)

    lines.append(separator)
    logger.info("\n".join(lines))

    # Persist comparison as JSON
    path = os.path.join(REPORT_DIR, "model_comparison.json")
    with open(path, "w") as f:
        json.dump(reports, f, indent=2)
    logger.info("Model comparison saved to: %s", path)
