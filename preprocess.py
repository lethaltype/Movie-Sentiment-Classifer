# =============================================================================
# preprocess.py — NLP Preprocessing Pipeline
# =============================================================================
"""
Handles:
  • Loading the IMDB dataset from Keras
  • Cleaning raw text (HTML, punctuation, stopwords)
  • Fitting a Keras Tokenizer on training data
  • Converting integer sequences back to text (for debugging)
  • Padding / truncating sequences to a fixed length
  • Saving & loading the fitted tokenizer
"""

import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from utils import clean_text, save_pickle, load_pickle, get_logger, set_seeds

logger = get_logger("preprocess")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _decode_imdb_review(encoded: list[int], word_index: dict) -> str:
    """Decode an IMDB integer sequence back to plain text."""
    reverse_index = {v + 3: k for k, v in word_index.items()}
    reverse_index[0] = "<PAD>"
    reverse_index[1] = "<START>"
    reverse_index[2] = "<UNK>"
    reverse_index[3] = "<UNUSED>"
    return " ".join(reverse_index.get(i, "?") for i in encoded)


# ── Public API ────────────────────────────────────────────────────────────────

def load_imdb_data(
    num_words: int = config.NUM_WORDS,
) -> tuple[tuple, tuple, dict]:
    """
    Load the IMDB dataset from Keras, decode each review to raw text,
    apply cleaning, and return numpy arrays ready for tokenization.

    Returns
    -------
    (X_train_text, y_train), (X_test_text, y_test), word_index
    """
    logger.info("Loading IMDB dataset (vocab=%d) …", num_words)
    (X_train_enc, y_train), (X_test_enc, y_test) = imdb.load_data(
        num_words=num_words,
        skip_top=config.IMDB_SKIP_TOP,
        seed=config.RANDOM_SEED,
    )
    word_index = imdb.get_word_index()

    logger.info("Decoding %d training reviews …", len(X_train_enc))
    X_train_text = np.array([
        clean_text(_decode_imdb_review(r, word_index)) for r in X_train_enc
    ])

    logger.info("Decoding %d test reviews …", len(X_test_enc))
    X_test_text = np.array([
        clean_text(_decode_imdb_review(r, word_index)) for r in X_test_enc
    ])

    logger.info(
        "Dataset loaded. Train: %d | Test: %d | Positive train: %.1f%%",
        len(X_train_text),
        len(X_test_text),
        np.mean(y_train) * 100,
    )
    return (X_train_text, y_train), (X_test_text, y_test), word_index


def build_tokenizer(
    texts: np.ndarray,
    num_words: int = config.NUM_WORDS,
    oov_token: str = config.OOV_TOKEN,
) -> Tokenizer:
    """
    Fit a Keras Tokenizer on the supplied texts.

    Parameters
    ----------
    texts     : 1-D array of cleaned review strings
    num_words : Maximum vocabulary size (keeps top-N by frequency)
    oov_token : Token to use for out-of-vocabulary words at inference

    Returns
    -------
    Fitted Tokenizer instance
    """
    logger.info("Fitting tokenizer on %d texts (vocab_size=%d) …", len(texts), num_words)
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    logger.info("Tokenizer fitted. Unique tokens found: %d", vocab_size)
    return tokenizer


def texts_to_padded_sequences(
    texts: np.ndarray,
    tokenizer: Tokenizer,
    max_len: int = config.MAX_SEQUENCE_LEN,
    padding: str = config.PADDING_TYPE,
    truncating: str = config.TRUNCATING_TYPE,
) -> np.ndarray:
    """
    Convert an array of text strings to a 2-D padded integer matrix.

    Parameters
    ----------
    texts      : Array of clean review strings
    tokenizer  : Fitted Keras Tokenizer
    max_len    : Target sequence length (pad/truncate to this)
    padding    : "pre" | "post"
    truncating : "pre" | "post"

    Returns
    -------
    np.ndarray of shape (len(texts), max_len)
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding=padding,
        truncating=truncating,
    )
    logger.info(
        "Sequences padded. Shape: %s  (padding=%s, truncating=%s)",
        padded.shape,
        padding,
        truncating,
    )
    return padded


def save_tokenizer(tokenizer: Tokenizer, path: str = config.TOKENIZER_PATH) -> None:
    """Persist the fitted tokenizer as a pickle file."""
    save_pickle(tokenizer, path)


def load_tokenizer(path: str = config.TOKENIZER_PATH) -> Tokenizer:
    """Reload the fitted tokenizer from disk."""
    return load_pickle(path)


def preprocess_single_review(
    review: str,
    tokenizer: Tokenizer,
    max_len: int = config.MAX_SEQUENCE_LEN,
) -> np.ndarray:
    """
    Convenience wrapper: clean → tokenise → pad a *single* review string.

    Returns
    -------
    np.ndarray of shape (1, max_len) — ready to feed directly into the model
    """
    cleaned = clean_text(review)
    return texts_to_padded_sequences(
        np.array([cleaned]),
        tokenizer,
        max_len=max_len,
    )


# ── CLI entry-point (run standalone to inspect preprocessing) ─────────────────
if __name__ == "__main__":
    set_seeds()
    (X_train, y_train), (X_test, y_test), _ = load_imdb_data()

    print("\n── Sample cleaned reviews ──────────────────────────────────────")
    for i in range(3):
        label = "POS" if y_train[i] == 1 else "NEG"
        print(f"[{label}] {X_train[i][:120]} …\n")

    tokenizer = build_tokenizer(X_train)
    X_train_pad = texts_to_padded_sequences(X_train, tokenizer)
    X_test_pad  = texts_to_padded_sequences(X_test,  tokenizer)

    print(f"\nX_train padded shape : {X_train_pad.shape}")
    print(f"X_test  padded shape : {X_test_pad.shape}")
    print(f"Vocabulary size      : {len(tokenizer.word_index) + 1}")
    print(f"\nFirst padded sequence (last 20 values): {X_train_pad[0][-20:]}")
