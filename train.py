# =============================================================================
# train.py — End-to-End Training Pipeline
# =============================================================================
"""
Usage
-----
    python train.py                          # default config
    python train.py --epochs 15 --batch 64  # override specific hyper-params

Workflow
--------
1. Set random seeds for reproducibility
2. Load & decode IMDB reviews → clean text
3. Fit a Keras Tokenizer on the training corpus
4. Pad/truncate all sequences to MAX_SEQUENCE_LEN
5. Build the LSTM model
6. Train with EarlyStopping + ReduceLROnPlateau callbacks
7. Evaluate on the held-out test set
8. Save the trained model and tokenizer to disk
9. Plot and save the training history curves
"""

import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

import config
from preprocess import (
    load_imdb_data,
    build_tokenizer,
    texts_to_padded_sequences,
    save_tokenizer,
)
from model import build_lstm_model, model_summary_str
from utils import (
    get_logger,
    plot_training_history,
    save_pickle,
    set_seeds,
    timer,
)

logger = get_logger("train")


# ── Argument parser ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Movie Sentiment LSTM Classifier"
    )
    parser.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    parser.add_argument("--batch",       type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lstm_units",  type=int,   default=config.LSTM_UNITS)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Wrap LSTM in a Bidirectional layer")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="Disable EarlyStopping callback")
    return parser.parse_args()


# ── Main training routine ─────────────────────────────────────────────────────
@timer
def train(args=None):
    if args is None:
        args = parse_args()

    set_seeds(config.RANDOM_SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  MOVIE REVIEW SENTIMENT CLASSIFIER 2026 — Training")
    logger.info("=" * 60)

    (X_train_text, y_train), (X_test_text, y_test), _ = load_imdb_data(
        num_words=config.NUM_WORDS
    )

    # ── 2. Tokenise ───────────────────────────────────────────────────────────
    tokenizer = build_tokenizer(X_train_text, num_words=config.NUM_WORDS)
    save_tokenizer(tokenizer, config.TOKENIZER_PATH)

    # ── 3. Pad sequences ──────────────────────────────────────────────────────
    X_train = texts_to_padded_sequences(X_train_text, tokenizer)
    X_test  = texts_to_padded_sequences(X_test_text,  tokenizer)

    logger.info("X_train: %s | y_train: %s", X_train.shape, y_train.shape)
    logger.info("X_test : %s | y_test : %s", X_test.shape,  y_test.shape)

    # ── 4. Build model ────────────────────────────────────────────────────────
    vocab_size = min(len(tokenizer.word_index) + 1, config.NUM_WORDS + 1)
    model = build_lstm_model(
        vocab_size=vocab_size,
        lstm_units=args.lstm_units,
        learning_rate=args.lr,
        bidirectional=args.bidirectional,
    )
    logger.info("\n%s", model_summary_str(model))

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, "tensorboard"),
            histogram_freq=1,
        ),
    ]
    if not args.no_early_stop:
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                patience=3,
                restore_best_weights=True,
                verbose=1,
            )
        )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    logger.info("Starting training — epochs=%d, batch=%d, lr=%.5f",
                args.epochs, args.batch, args.lr)

    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Final evaluation on held-out test set ──────────────────────────────
    logger.info("Evaluating on test set …")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test  Loss    : %.4f", test_loss)
    logger.info("Test  Accuracy: %.4f (%.2f%%)", test_acc, test_acc * 100)

    # ── 8. Persist artefacts ──────────────────────────────────────────────────
    model.save(config.MODEL_PATH)
    logger.info("Model saved → %s", config.MODEL_PATH)

    save_pickle(history.history, config.HISTORY_PATH)
    logger.info("History saved → %s", config.HISTORY_PATH)

    # ── 9. Plot curves ────────────────────────────────────────────────────────
    plot_path = os.path.join(config.ASSETS_DIR, "training_history.png")
    plot_training_history(history, save_path=plot_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info("  Best val accuracy : %.4f", max(history.history["val_accuracy"]))
    logger.info("  Test accuracy     : %.4f", test_acc)
    logger.info("  Model saved to    : %s", config.MODEL_PATH)
    logger.info("=" * 60)

    return model, tokenizer, history


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
