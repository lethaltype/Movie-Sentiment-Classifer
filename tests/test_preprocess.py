# =============================================================================
# tests/test_preprocess.py — Unit Tests for Preprocessing
# =============================================================================
"""
Run with:  pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import clean_text
from preprocess import build_tokenizer, texts_to_padded_sequences, preprocess_single_review
import config


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello WORLD") == clean_text("hello world")

    def test_strips_html(self):
        result = clean_text("<br />Great movie!", remove_stopwords=False)
        assert "<" not in result and ">" not in result

    def test_strips_url(self):
        result = clean_text("Visit https://example.com for details", remove_stopwords=False)
        assert "http" not in result

    def test_removes_punctuation(self):
        result = clean_text("Wow!!! This is... amazing?", remove_stopwords=False)
        assert "!" not in result and "." not in result and "?" not in result

    def test_stopword_removal(self):
        result_with    = clean_text("This is a good movie", remove_stopwords=True)
        result_without = clean_text("This is a good movie", remove_stopwords=False)
        # "this", "is", "a" are stopwords → shorter with removal
        assert len(result_with.split()) < len(result_without.split())

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_returns_string(self):
        assert isinstance(clean_text("any text"), str)


# ── build_tokenizer ───────────────────────────────────────────────────────────

class TestBuildTokenizer:
    def setup_method(self):
        self.corpus = np.array([
            "great movie loved acting",
            "terrible film horrible waste time",
            "average movie nothing special",
        ])

    def test_returns_tokenizer(self):
        tok = build_tokenizer(self.corpus, num_words=100)
        assert isinstance(tok, Tokenizer)

    def test_word_index_not_empty(self):
        tok = build_tokenizer(self.corpus, num_words=100)
        assert len(tok.word_index) > 0

    def test_oov_token_present(self):
        tok = build_tokenizer(self.corpus, num_words=100, oov_token="<OOV>")
        assert "<OOV>" in tok.word_index


# ── texts_to_padded_sequences ─────────────────────────────────────────────────

class TestPadding:
    def setup_method(self):
        corpus = np.array(["great movie", "terrible horrible waste time money bad"])
        self.tok = build_tokenizer(corpus, num_words=50)

    def test_output_shape(self):
        texts = np.array(["great film", "waste money"])
        padded = texts_to_padded_sequences(texts, self.tok, max_len=10)
        assert padded.shape == (2, 10)

    def test_dtype_integer(self):
        texts  = np.array(["great film"])
        padded = texts_to_padded_sequences(texts, self.tok, max_len=10)
        assert np.issubdtype(padded.dtype, np.integer)

    def test_short_review_padded_with_zeros(self):
        texts  = np.array(["great"])
        padded = texts_to_padded_sequences(texts, self.tok, max_len=5, padding="pre")
        # pre-padding means zeros at the start
        assert padded[0, 0] == 0


# ── preprocess_single_review ──────────────────────────────────────────────────

class TestSingleReview:
    def setup_method(self):
        corpus = np.array(["great movie loved acting", "terrible film horrible"])
        self.tok = build_tokenizer(corpus, num_words=50)

    def test_output_shape(self):
        seq = preprocess_single_review("A great movie!", self.tok, max_len=20)
        assert seq.shape == (1, 20)

    def test_handles_html(self):
        seq = preprocess_single_review("<b>Great</b> movie!", self.tok, max_len=20)
        assert seq.shape == (1, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
