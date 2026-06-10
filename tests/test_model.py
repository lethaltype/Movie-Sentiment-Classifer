# =============================================================================
# tests/test_model.py — Unit Tests for Model Architecture
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from tensorflow.keras import Model

from model import build_lstm_model
import config


class TestBuildLSTMModel:
    def test_returns_keras_model(self):
        model = build_lstm_model(vocab_size=1000, max_len=50)
        assert isinstance(model, Model)

    def test_output_shape(self):
        model  = build_lstm_model(vocab_size=1000, max_len=50)
        dummy  = np.zeros((4, 50), dtype=np.int32)
        output = model.predict(dummy, verbose=0)
        assert output.shape == (4, 1)

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        model  = build_lstm_model(vocab_size=1000, max_len=50)
        dummy  = np.random.randint(0, 1000, (10, 50))
        output = model.predict(dummy, verbose=0).flatten()
        assert np.all(output >= 0) and np.all(output <= 1)

    def test_bidirectional_variant(self):
        model = build_lstm_model(vocab_size=1000, max_len=50, bidirectional=True)
        # Should have a Bidirectional layer somewhere
        layer_names = [l.name for l in model.layers]
        assert any("bidirectional" in n.lower() for n in layer_names)

    def test_has_embedding_layer(self):
        model = build_lstm_model(vocab_size=1000, max_len=50)
        layer_names = [l.name for l in model.layers]
        assert "word_embedding" in layer_names

    def test_parameter_count_positive(self):
        model = build_lstm_model(vocab_size=1000, max_len=50)
        assert model.count_params() > 0

    def test_custom_learning_rate(self):
        """Verify model compiles without error with custom LR."""
        model = build_lstm_model(vocab_size=500, max_len=30, learning_rate=1e-4)
        assert model.optimizer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
