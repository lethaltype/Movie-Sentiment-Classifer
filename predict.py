# =============================================================================
# predict.py — Inference / Prediction Script
# =============================================================================
"""
Accepts custom movie review text and predicts sentiment (Positive / Negative).

Usage
-----
  Single review (CLI flag):
      python predict.py --review "This movie was absolutely incredible!"

  Interactive REPL mode (no args):
      python predict.py

  Batch mode (file of reviews, one per line):
      python predict.py --file reviews.txt

  JSON output (useful for downstream integrations):
      python predict.py --review "Amazing film!" --json
"""

import argparse
import json
import os
import sys
import numpy as np

import config
from preprocess import preprocess_single_review, load_tokenizer
from save_model import load_keras_model
from utils import get_logger, clean_text

logger = get_logger("predict")


# ── Core prediction function ──────────────────────────────────────────────────

def predict_sentiment(
    review: str,
    model,
    tokenizer,
    threshold: float = config.POSITIVE_THRESHOLD,
) -> dict:
    """
    Predict the sentiment of a single movie review.

    Parameters
    ----------
    review    : Raw review text (HTML, punctuation, casing all OK)
    model     : Loaded Keras model
    tokenizer : Fitted Keras Tokenizer
    threshold : Decision boundary for Positive (default 0.5)

    Returns
    -------
    dict with keys: review, cleaned_review, probability, label, confidence
    """
    if not review.strip():
        raise ValueError("Review text cannot be empty.")

    cleaned = clean_text(review)
    seq     = preprocess_single_review(review, tokenizer)
    prob    = float(model.predict(seq, verbose=0)[0][0])
    label   = config.LABEL_MAP[int(prob >= threshold)]

    # Confidence = how far the sigmoid output is from the decision boundary
    confidence = abs(prob - 0.5) * 2   # 0 = uncertain, 1 = maximally confident

    return {
        "review":         review[:200] + ("…" if len(review) > 200 else ""),
        "cleaned_review": cleaned[:200],
        "probability":    round(prob, 6),
        "label":          label,
        "confidence":     round(confidence, 4),
    }


def predict_batch(reviews: list[str], model, tokenizer) -> list[dict]:
    """Run predict_sentiment on a list of review strings."""
    results = []
    for r in reviews:
        try:
            results.append(predict_sentiment(r, model, tokenizer))
        except ValueError as e:
            results.append({"review": r, "error": str(e)})
    return results


# ── Pretty-printing ───────────────────────────────────────────────────────────

def _print_result(result: dict, idx: int = None) -> None:
    prefix = f"[{idx}] " if idx is not None else ""
    print("\n" + "─" * 60)
    if "error" in result:
        print(f"{prefix}⚠️  Error: {result['error']}")
        return
    prob       = result["probability"]
    label      = result["label"]
    confidence = result["confidence"]

    bar_len  = 40
    filled   = round(prob * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(f"{prefix}Review   : {result['review']}")
    print(f"Cleaned  : {result['cleaned_review'][:80]}…")
    print(f"Label    : {label}")
    print(f"P(pos)   : {prob:.4f}  [{bar}]")
    print(f"Confidence: {confidence * 100:.1f}%")
    print("─" * 60)


# ── Interactive REPL ──────────────────────────────────────────────────────────

def interactive_mode(model, tokenizer) -> None:
    print("\n" + "=" * 60)
    print("  🎬  Movie Review Sentiment Classifier 2026")
    print("  Type a review and press Enter to classify.")
    print("  Type  'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            review = input("📝 Enter review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if review.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 👋")
            break
        if not review:
            print("⚠️  Please enter some text.")
            continue

        result = predict_sentiment(review, model, tokenizer)
        _print_result(result)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Predict sentiment of movie reviews",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--review", "-r", type=str, default=None,
        help="A single review string to classify",
    )
    p.add_argument(
        "--file", "-f", type=str, default=None,
        help="Path to a text file with one review per line",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of pretty-print",
    )
    p.add_argument(
        "--threshold", type=float, default=config.POSITIVE_THRESHOLD,
        help=f"Decision threshold (default: {config.POSITIVE_THRESHOLD})",
    )
    p.add_argument(
        "--model_path", default=config.MODEL_PATH,
        help="Path to the .keras model file",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load model & tokenizer ─────────────────────────────────────────────
    logger.info("Loading model and tokenizer …")
    model     = load_keras_model(args.model_path)
    tokenizer = load_tokenizer(config.TOKENIZER_PATH)
    logger.info("Ready.")

    # ── Single review ──────────────────────────────────────────────────────
    if args.review:
        result = predict_sentiment(args.review, model, tokenizer, args.threshold)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_result(result)
        return

    # ── Batch from file ────────────────────────────────────────────────────
    if args.file:
        if not os.path.exists(args.file):
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        with open(args.file, encoding="utf-8") as fh:
            reviews = [line.strip() for line in fh if line.strip()]
        logger.info("Read %d reviews from %s", len(reviews), args.file)
        results = predict_batch(reviews, model, tokenizer)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for i, res in enumerate(results, start=1):
                _print_result(res, idx=i)
        return

    # ── Interactive mode (no args) ─────────────────────────────────────────
    interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
