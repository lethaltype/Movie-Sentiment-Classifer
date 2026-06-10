# =============================================================================
# model.py — LSTM Model Architecture
# =============================================================================
"""
Defines the LSTM-based binary sentiment classifier.

Architecture
------------
Input (integer token IDs, shape: [batch, MAX_SEQUENCE_LEN])
  │
  ▼
Embedding(NUM_WORDS, EMBEDDING_DIM)           ← learns word vectors during training
  │
  ▼
SpatialDropout1D(0.3)                          ← regularise embedding outputs
  │
  ▼
LSTM(LSTM_UNITS, dropout=…, recurrent_dropout=…)
  │
  ▼
Dense(64, activation="relu")                   ← classification head
  │
  ▼
Dropout(DROPOUT_RATE)
  │
  ▼
Dense(1, activation="sigmoid")                 ← output probability [0, 1]

Loss    : BinaryCrossentropy
Optimiser: Adam
Metric  : Accuracy
"""

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Embedding,
    SpatialDropout1D,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.optimizers import Adam

import config
from utils import get_logger

logger = get_logger("model")


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_lstm_model(
    vocab_size: int = config.NUM_WORDS + 1,
    embedding_dim: int = config.EMBEDDING_DIM,
    max_len: int = config.MAX_SEQUENCE_LEN,
    lstm_units: int = config.LSTM_UNITS,
    dropout_rate: float = config.DROPOUT_RATE,
    recurrent_dropout: float = config.RECURRENT_DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
    bidirectional: bool = False,
) -> Model:
    """
    Build and compile the sentiment-classifier LSTM.

    Parameters
    ----------
    vocab_size         : Number of unique tokens (vocabulary size + 1 for padding)
    embedding_dim      : Dense vector size for each token
    max_len            : Fixed input sequence length
    lstm_units         : Number of LSTM memory cells
    dropout_rate       : Dropout applied to Dense layer
    recurrent_dropout  : Dropout applied inside LSTM recurrent connections
    learning_rate      : Adam learning rate
    bidirectional      : Wrap LSTM in a Bidirectional layer if True

    Returns
    -------
    Compiled tf.keras.Model
    """
    inputs = Input(shape=(max_len,), name="token_ids")

    # Embedding layer — converts integer tokens → dense vectors
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="word_embedding",
    )(inputs)

    # Spatial dropout on the embedding (drops entire tokens rather than individual dims)
    x = SpatialDropout1D(0.3, name="embedding_dropout")(x)

    # Recurrent layer
    lstm_layer = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout,
        name="lstm",
    )
    if bidirectional:
        x = Bidirectional(lstm_layer, name="bi_lstm")(x)
    else:
        x = lstm_layer(x)

    # Classification head
    x = Dense(64, activation="relu", name="dense_relu")(x)
    x = Dropout(dropout_rate, name="dense_dropout")(x)
    outputs = Dense(1, activation="sigmoid", name="sentiment_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SentimentLSTM")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Model built successfully.")
    logger.info("  Vocab size   : %d", vocab_size)
    logger.info("  Embedding dim: %d", embedding_dim)
    logger.info("  LSTM units   : %d", lstm_units)
    logger.info("  Bidirectional: %s", bidirectional)
    logger.info("  Max seq len  : %d", max_len)

    return model


def model_summary_str(model: Model) -> str:
    """Return the model summary as a plain string (for logging)."""
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_lstm_model()
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
