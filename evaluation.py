# =============================================================================
# evaluation.py — Comprehensive Model Evaluation
# =============================================================================
"""
Loads the trained model + tokenizer, runs inference on the full IMDB test set,
and produces:
  • Classification Report (precision, recall, F1, support)
  • Confusion Matrix (numeric + heatmap saved to assets/)
  • ROC Curve + AUC score
  • Precision-Recall Curve
  • Top wrong predictions (to understand failure modes)
  • Training history curves (replotted from saved history)

Usage
-----
    python evaluation.py
    python evaluation.py --batch 256   # larger batch for faster inference
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

import config
from preprocess import load_imdb_data, build_tokenizer, texts_to_padded_sequences, load_tokenizer
from save_model import load_keras_model
from utils import get_logger, load_pickle, plot_training_history, plot_confusion_matrix, set_seeds

logger = get_logger("evaluation")


# ── Helpers ───────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, save_path: str = None) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="gray", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Sentiment Classifier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved → %s", save_path)
    plt.close()
    return auc


def plot_precision_recall(y_true, y_prob, save_path: str = None) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.step(recall, precision, where="post", lw=2, color="#4CAF50",
            label=f"PR Curve (AP = {ap:.4f})")
    ax.fill_between(recall, precision, step="post", alpha=0.15, color="#4CAF50")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Sentiment Classifier",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve saved → %s", save_path)
    plt.close()
    return ap


def show_worst_predictions(
    reviews: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n: int = 5,
) -> None:
    """Print the N most confident wrong predictions."""
    y_pred = (y_prob >= config.POSITIVE_THRESHOLD).astype(int)
    wrong_mask = y_pred != y_true

    if not wrong_mask.any():
        logger.info("No wrong predictions found!")
        return

    errors_idx   = np.where(wrong_mask)[0]
    # Confidence = how far the probability is from 0.5
    confidence   = np.abs(y_prob[errors_idx] - 0.5)
    top_n        = errors_idx[np.argsort(-confidence)[:n]]

    logger.info("─" * 60)
    logger.info("Top-%d Most Confident Wrong Predictions", n)
    logger.info("─" * 60)
    for rank, idx in enumerate(top_n, start=1):
        true_label = config.LABEL_MAP[y_true[idx]]
        pred_label = config.LABEL_MAP[y_pred[idx]]
        prob       = y_prob[idx]
        snippet    = reviews[idx][:120]
        logger.info(
            "[%d] True=%-12s Pred=%-12s P(pos)=%.4f\n    Review: %s…\n",
            rank, true_label, pred_label, prob, snippet,
        )


# ── Main evaluation routine ───────────────────────────────────────────────────

def evaluate(batch_size: int = config.BATCH_SIZE) -> dict:
    set_seeds()

    # 1. Load data
    (X_train_text, y_train), (X_test_text, y_test), _ = load_imdb_data()

    # 2. Tokenize (use the saved tokenizer, don't refit)
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    X_test_pad = texts_to_padded_sequences(X_test_text, tokenizer)

    # 3. Load model
    model = load_keras_model(config.MODEL_PATH)

    # 4. Predict
    logger.info("Running inference on %d test samples …", len(X_test_pad))
    y_prob = model.predict(X_test_pad, batch_size=batch_size, verbose=1).flatten()
    y_pred = (y_prob >= config.POSITIVE_THRESHOLD).astype(int)

    # 5. Metrics
    logger.info("\n%s\n%s\n%s",
        "=" * 60,
        "  CLASSIFICATION REPORT",
        "=" * 60,
    )
    report = classification_report(
        y_test, y_pred,
        target_names=["Negative", "Positive"],
        digits=4,
    )
    print(report)

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:\n%s", cm)
    plot_confusion_matrix(cm, save_path=os.path.join(config.ASSETS_DIR, "confusion_matrix.png"))

    # 7. ROC + AUC
    auc = plot_roc_curve(y_test, y_prob,
                         save_path=os.path.join(config.ASSETS_DIR, "roc_curve.png"))
    logger.info("ROC-AUC: %.4f", auc)

    # 8. Precision-Recall
    ap = plot_precision_recall(y_test, y_prob,
                               save_path=os.path.join(config.ASSETS_DIR, "pr_curve.png"))
    logger.info("Average Precision: %.4f", ap)

    # 9. Training history curves (if history file exists)
    if os.path.exists(config.HISTORY_PATH):
        history = load_pickle(config.HISTORY_PATH)
        plot_training_history(history,
                              save_path=os.path.join(config.ASSETS_DIR, "training_history.png"))

    # 10. Worst predictions
    show_worst_predictions(X_test_text, y_test, y_prob, n=5)

    # 11. Return metrics dict
    acc = float((y_pred == y_test).mean())
    metrics = {
        "test_accuracy":      acc,
        "roc_auc":            auc,
        "average_precision":  ap,
    }
    logger.info("=" * 60)
    logger.info("  Evaluation Summary")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info("  %-22s : %.4f", k, v)
    logger.info("=" * 60)

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained sentiment model")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(batch_size=args.batch)
