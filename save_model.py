# =============================================================================
# save_model.py — Save, Load & Export Utilities
# =============================================================================
"""
Provides convenience functions to:
  • Save / load the Keras model in the native .keras format
  • Export the model to TensorFlow SavedModel format
  • Export to ONNX (requires tf2onnx; optional)
  • Verify a saved model produces correct predictions
  • Print a model card summary

Usage
-----
    python save_model.py          # load and verify the saved model
    python save_model.py --export # also export to SavedModel format
"""

import os
import argparse
import numpy as np
import tensorflow as tf

import config
from preprocess import load_tokenizer, preprocess_single_review
from utils import get_logger

logger = get_logger("save_model")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_keras_model(model: tf.keras.Model, path: str = config.MODEL_PATH) -> None:
    """Save the model in Keras v3 native format (.keras)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    size_mb = os.path.getsize(path) / 1_048_576
    logger.info("Keras model saved → %s  (%.2f MB)", path, size_mb)


def save_savedmodel(model: tf.keras.Model, export_dir: str) -> None:
    """Export to TensorFlow SavedModel format (useful for TF Serving)."""
    os.makedirs(export_dir, exist_ok=True)
    tf.saved_model.save(model, export_dir)
    logger.info("SavedModel exported → %s", export_dir)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_keras_model(path: str = config.MODEL_PATH) -> tf.keras.Model:
    """
    Load a Keras model from disk.

    Returns
    -------
    Compiled tf.keras.Model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No model found at '{path}'.\n"
            "Please run  python train.py  first."
        )
    model = tf.keras.models.load_model(path)
    logger.info("Model loaded ← %s", path)
    return model


# ── Verify ────────────────────────────────────────────────────────────────────

def verify_model(model: tf.keras.Model, tokenizer) -> None:
    """Run a small smoke-test on the loaded model with known-sentiment reviews."""
    test_cases = [
        ("This movie was absolutely brilliant! A masterpiece of storytelling.", "Positive"),
        ("Terrible film. Waste of time and money. Do not watch.", "Negative"),
        ("The acting was mediocre but the cinematography saved it.", "Mixed → likely Negative"),
        ("One of the greatest films I have ever seen!", "Positive"),
    ]

    logger.info("=" * 60)
    logger.info("  Model Verification Smoke-Test")
    logger.info("=" * 60)

    all_pass = True
    for review, expected in test_cases:
        seq = preprocess_single_review(review, tokenizer)
        prob = float(model.predict(seq, verbose=0)[0][0])
        label = config.LABEL_MAP[int(prob >= config.POSITIVE_THRESHOLD)]
        status = "✅" if expected.split()[0] in label else "⚠️ "
        if "⚠️" in status:
            all_pass = False
        logger.info("%s  P(pos)=%.4f  Predicted: %-12s  Review: %.60s…",
                    status, prob, label, review)

    logger.info("=" * 60)
    logger.info("Smoke-test %s", "PASSED" if all_pass else "completed with warnings")


# ── Model Card ────────────────────────────────────────────────────────────────

def print_model_card(model: tf.keras.Model) -> None:
    """Print a concise model card to stdout."""
    card = f"""
╔══════════════════════════════════════════════════════════════╗
║          MOVIE REVIEW SENTIMENT CLASSIFIER 2026              ║
║                     MODEL CARD                               ║
╠══════════════════════════════════════════════════════════════╣
║  Architecture  : LSTM (Bidirectional optional)               ║
║  Framework     : TensorFlow / Keras                          ║
║  Dataset       : IMDB (50 000 reviews, balanced)             ║
║  Vocabulary    : {config.NUM_WORDS:,} tokens                           ║
║  Sequence len  : {config.MAX_SEQUENCE_LEN} tokens                              ║
║  Embedding dim : {config.EMBEDDING_DIM}                                      ║
║  LSTM units    : {config.LSTM_UNITS}                                       ║
║  Parameters    : {model.count_params():,}                         ║
║  Task          : Binary Sentiment Classification             ║
║  Labels        : Positive / Negative                         ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(card)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Save/Load/Verify the sentiment model")
    parser.add_argument("--model_path", default=config.MODEL_PATH,
                        help="Path to the .keras model file")
    parser.add_argument("--export", action="store_true",
                        help="Also export to TensorFlow SavedModel format")
    parser.add_argument("--export_dir", default=os.path.join(config.MODEL_DIR, "saved_model"),
                        help="Directory for SavedModel export")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_keras_model(args.model_path)
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)

    print_model_card(model)
    verify_model(model, tokenizer)

    if args.export:
        save_savedmodel(model, args.export_dir)
        logger.info("Export complete → %s", args.export_dir)
