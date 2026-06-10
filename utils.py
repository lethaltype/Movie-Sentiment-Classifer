# =============================================================================
# utils.py — Shared Utility Functions
# =============================================================================

import os
import re
import pickle
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers)

import config

# ── Logger ────────────────────────────────────────────────────────────────────
def get_logger(name: str = "sentiment") -> logging.Logger:
    """Return a configured logger that writes to stdout and a log file."""
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger
    logger.setLevel(config.LOG_LEVEL)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(config.LOG_DIR, "run.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = get_logger()


# ── Text Cleaning ─────────────────────────────────────────────────────────────
# Common English stopwords (lightweight — avoids requiring NLTK download)
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through","during",
    "before","after","above","below","to","from","up","down","in","out","on",
    "off","over","under","again","further","then","once","here","there","when",
    "where","why","how","all","both","each","few","more","most","other","some",
    "such","no","nor","not","only","own","same","so","than","too","very","s",
    "t","can","will","just","don","should","now","d","ll","m","o","re","ve",
    "y","ain","aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma",
    "mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
}


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Clean raw review text:
      1. Lowercase
      2. Remove HTML tags
      3. Remove URLs
      4. Keep only alphabetic characters + spaces
      5. Collapse whitespace
      6. Optionally remove stopwords
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)            # strip HTML
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)           # keep letters only
    text = re.sub(r"\s+", " ", text).strip()         # collapse spaces

    if remove_stopwords:
        tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        text = " ".join(tokens)

    return text


# ── Serialisation helpers ─────────────────────────────────────────────────────
def save_pickle(obj, path: str) -> None:
    """Pickle an object to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved → %s", path)


def load_pickle(path: str):
    """Load a pickled object from *path*."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded ← %s", path)
    return obj


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training_history(history, save_path: str = None) -> None:
    """Plot accuracy & loss curves from a Keras History object (or dict)."""
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history  # already a dict

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=15, fontweight="bold")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(hist["accuracy"],     label="Train Accuracy",      linewidth=2, color="#2196F3")
    ax.plot(hist["val_accuracy"], label="Validation Accuracy", linewidth=2, color="#FF5722", linestyle="--")
    ax.set_title("Accuracy over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(hist["loss"],     label="Train Loss",      linewidth=2, color="#4CAF50")
    ax.plot(hist["val_loss"], label="Validation Loss", linewidth=2, color="#F44336", linestyle="--")
    ax.set_title("Loss over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved → %s", save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> None:
    """Render a 2×2 confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved → %s", save_path)
    else:
        plt.show()
    plt.close()


# ── Misc ──────────────────────────────────────────────────────────────────────
def timer(func):
    """Decorator that logs execution time."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info("⏱  %s completed in %.2f s", func.__name__, elapsed)
        return result
    return wrapper


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Pin all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
