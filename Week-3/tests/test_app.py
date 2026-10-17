"""
Unit and integration tests for Week 3 Text Generation System and model_utils.
"""

import os
import sys
import tempfile
import json
import pytest
import numpy as np

# Ensure Week-3 is on sys.path
WEEK3_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WEEK3_DIR not in sys.path:
    sys.path.insert(0, WEEK3_DIR)

from model_utils import (
    build_char_lstm_model,
    build_vocabulary,
    encode_seed_text,
    sample_next_char,
    generate_text,
    save_artifacts,
    load_artifacts,
)


class TestVocabularyAndEncoding:
    @pytest.fixture
    def sample_text(self):
        return "Hello World! To be or not to be."

    def test_build_vocabulary(self, sample_text):
        unique_chars, char_to_idx, idx_to_char = build_vocabulary(sample_text)
        assert len(unique_chars) == len(char_to_idx)
        assert len(char_to_idx) == len(idx_to_char)
        assert "H" in char_to_idx
        assert " " in char_to_idx
        # Check bidirectional consistency
        for ch, idx in char_to_idx.items():
            assert idx_to_char[idx] == ch

    def test_encode_seed_text_padding(self):
        char_to_idx = {"a": 0, "b": 1, " ": 2}
        seq_len = 10
        seed = "ab"
        encoded = encode_seed_text(seed, char_to_idx, seq_length=seq_len)
        assert len(encoded) == seq_len
        # Expect padding spaces at start
        pad_idx = char_to_idx[" "]
        assert encoded[:8] == [pad_idx] * 8
        assert encoded[8:] == [0, 1]

    def test_encode_seed_text_truncation(self):
        char_to_idx = {"a": 0, "b": 1, "c": 2}
        seq_len = 3
        seed = "abcba"
        encoded = encode_seed_text(seed, char_to_idx, seq_length=seq_len)
        assert len(encoded) == seq_len
        # Expect last 3 characters 'cba' -> [2, 1, 0]
        assert encoded == [2, 1, 0]

    def test_unknown_character_handling(self):
        char_to_idx = {"a": 0, "b": 1, " ": 2}
        seq_len = 5
        seed = "a@#b"  # '@' and '#' not in vocabulary
        encoded = encode_seed_text(seed, char_to_idx, seq_length=seq_len, unknown_char_strategy="ignore")
        assert len(encoded) == seq_len
        # Valid characters 'ab' are preserved
        assert encoded[-2:] == [0, 1]


class TestTemperatureSampling:
    def test_deterministic_greedy_at_zero_temperature(self):
        preds = np.array([0.1, 0.7, 0.2])
        # Temperature close to 0 should always pick highest probability index (index 1)
        sampled = sample_next_char(preds, temperature=0.001)
        assert sampled == 1

    def test_low_temperature_peaks_mode(self):
        preds = np.array([0.05, 0.85, 0.10])
        samples = [sample_next_char(preds, temperature=0.2) for _ in range(50)]
        # At T=0.2, the 0.85 prob becomes overwhelmingly dominant (>99%)
        assert samples.count(1) >= 45

    def test_high_temperature_increases_entropy(self):
        preds = np.array([0.01, 0.98, 0.01])
        # High temperature flattens distribution
        samples_high = [sample_next_char(preds, temperature=1.5) for _ in range(100)]
        # At T=1.5, smaller probability classes are sampled noticeably more often
        sampled_non_modal = [s for s in samples_high if s != 1]
        assert len(sampled_non_modal) > 0


class TestModelArchitectureAndGeneration:
    @pytest.fixture
    def toy_model(self):
        model = build_char_lstm_model(
            vocab_size=10,
            seq_length=5,
            embedding_dim=16,
            lstm_units=32,
            dropout_rate=0.0,
        )
        return model

    def test_model_output_shape(self, toy_model):
        dummy_input = np.zeros((2, 5), dtype=np.int32)
        out = toy_model.predict(dummy_input, verbose=0)
        assert out.shape == (2, 10)
        # Verify valid probabilities summing to ~1
        np.testing.assert_allclose(np.sum(out, axis=1), [1.0, 1.0], atol=1e-4)

    def test_generate_text_length(self, toy_model):
        vocab = [chr(c) for c in range(ord('a'), ord('a') + 10)]
        char_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_char = {i: ch for i, ch in enumerate(vocab)}

        generated = generate_text(
            model=toy_model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seq_length=5,
            seed_text="abc",
            gen_length=25,
            temperature=0.8,
        )
        assert len(generated) == 25
        assert all(ch in vocab for ch in generated)


class TestArtifactSerialization:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            unique_chars = ["a", "b", "c", " "]
            vocab_size = len(unique_chars)
            model = build_char_lstm_model(
                vocab_size=vocab_size,
                seq_length=4,
                embedding_dim=8,
                lstm_units=16,
            )

            save_artifacts(
                models_dir=tmp_dir,
                model=model,
                unique_chars=unique_chars,
                seq_length=4,
                additional_config={"test_meta": 123},
            )

            # Files must exist
            assert os.path.exists(os.path.join(tmp_dir, "text_generator.keras"))
            assert os.path.exists(os.path.join(tmp_dir, "vocabulary.json"))
            assert os.path.exists(os.path.join(tmp_dir, "model_config.json"))

            # Load back
            loaded_model, c2i, i2c, cfg = load_artifacts(tmp_dir)
            assert len(c2i) == vocab_size
            assert len(i2c) == vocab_size
            assert cfg["seq_length"] == 4
            assert cfg["test_meta"] == 123


class TestRealModelIntegration:
    @pytest.fixture
    def models_dir(self):
        return os.path.join(WEEK3_DIR, "models")

    def test_saved_artifacts_exist(self, models_dir):
        assert os.path.exists(os.path.join(models_dir, "text_generator.keras"))
        assert os.path.exists(os.path.join(models_dir, "vocabulary.json"))
        assert os.path.exists(os.path.join(models_dir, "model_config.json"))

    def test_real_model_loading_and_inference(self, models_dir):
        model, char_to_idx, idx_to_char, config = load_artifacts(models_dir)
        assert len(char_to_idx) == 61
        assert config["seq_length"] == 40
        assert config["embedding_dim"] == 64
        assert config["lstm_units"] == 128

        seed = "First Citizen:\n"
        generated = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seq_length=config["seq_length"],
            seed_text=seed,
            gen_length=30,
            temperature=0.8,
        )
        assert len(generated) == 30
        assert all(ch in char_to_idx for ch in generated)

    def test_temperature_controls_real_model_entropy(self, models_dir):
        model, char_to_idx, idx_to_char, config = load_artifacts(models_dir)
        seed = "To be, or not to be"

        # Generate with low temperature
        gen_low = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seq_length=config["seq_length"],
            seed_text=seed,
            gen_length=50,
            temperature=0.1,
        )
        # Generate with high temperature
        gen_high = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seq_length=config["seq_length"],
            seed_text=seed,
            gen_length=50,
            temperature=1.4,
        )
        assert len(gen_low) == 50
        assert len(gen_high) == 50
        # High temperature should yield higher character entropy / distinct characters
        assert len(set(gen_high)) >= len(set(gen_low))

