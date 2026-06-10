# =============================================================================
# config.py — Central Configuration for Movie Review Sentiment Classifier 2026
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

MODEL_PATH      = os.path.join(MODEL_DIR, "sentiment_lstm.keras")
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "tokenizer.pkl")
HISTORY_PATH    = os.path.join(MODEL_DIR, "training_history.pkl")

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_WORDS       = 20_000     # Vocabulary size (top-N most frequent words)
IMDB_SKIP_TOP   = 0          # Skip the top-N most common words (can reduce noise)

# ── Preprocessing ─────────────────────────────────────────────────────────────
MAX_SEQUENCE_LEN = 300       # Pad / truncate every review to this many tokens
PADDING_TYPE     = "pre"     # Padding placement: "pre" or "post"
TRUNCATING_TYPE  = "pre"     # Truncation placement: "pre" or "post"
OOV_TOKEN        = "<OOV>"   # Token for out-of-vocabulary words

# ── Model Architecture ────────────────────────────────────────────────────────
EMBEDDING_DIM    = 128       # Dimensionality of the word-embedding layer
LSTM_UNITS       = 64        # Number of LSTM units in the recurrent layer
DROPOUT_RATE     = 0.5       # Dropout rate for regularization
RECURRENT_DROPOUT= 0.2       # Recurrent dropout inside the LSTM cell

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 128
EPOCHS           = 10
VALIDATION_SPLIT = 0.2       # Fraction of training data used for validation
LEARNING_RATE    = 1e-3      # Adam optimizer learning rate
RANDOM_SEED      = 42

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_MAP        = {0: "Negative ❌", 1: "Positive ✅"}
POSITIVE_THRESHOLD = 0.5     # Sigmoid output ≥ this → Positive

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL        = "INFO"

# ── Create dirs if they don't exist ──────────────────────────────────────────
for _dir in (MODEL_DIR, DATA_DIR, LOG_DIR, ASSETS_DIR):
    os.makedirs(_dir, exist_ok=True)
