"""
model_utils.py
==============
Reusable utilities for character-level LSTM text generation:
- Architecture construction
- Vocabulary extraction and serialization
- Sequence vectorization and padding
- Autoregressive text generation with numerically stable temperature sampling
- Artifact persistence and validation
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_char_lstm_model(
    vocab_size: int,
    seq_length: int = 50,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.005,
) -> keras.Model:
    """Build and compile a character-level LSTM language model."""
    inputs = layers.Input(shape=(seq_length,), dtype=tf.int32, name="char_input")
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="char_embedding",
    )(inputs)
    x = layers.LSTM(lstm_units, return_sequences=False, name="lstm_layer")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name="lstm_dropout")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", name="vocab_dense")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CharLSTMGenerator")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_vocabulary(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Extract sorted unique characters and construct bidirectional mapping dictionaries."""
    unique_chars = sorted(list(set(text)))
    char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx_to_char = {idx: ch for idx, ch in enumerate(unique_chars)}
    return unique_chars, char_to_idx, idx_to_char


def save_artifacts(
    models_dir: str,
    model: keras.Model,
    unique_chars: List[str],
    seq_length: int,
    additional_config: Optional[dict] = None,
) -> None:
    """Save trained model, vocabulary mappings, and configuration to disk."""
    os.makedirs(models_dir, exist_ok=True)

    # 1. Save Keras model
    model_path = os.path.join(models_dir, "text_generator.keras")
    model.save(model_path)

    # 2. Save vocabulary
    char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx_to_char = {str(idx): ch for idx, ch in enumerate(unique_chars)}
    vocab_payload = {
        "unique_chars": unique_chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(unique_chars),
    }
    with open(os.path.join(models_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, indent=2, ensure_ascii=False)

    # 3. Save config
    config_payload = {
        "seq_length": seq_length,
        "vocab_size": len(unique_chars),
        "model_file": "text_generator.keras",
    }
    if additional_config:
        config_payload.update(additional_config)
    with open(os.path.join(models_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


def load_artifacts(models_dir: str) -> Tuple[keras.Model, Dict[str, int], Dict[int, str], dict]:
    """Load model, vocabulary, and config from the specified directory."""
    model_path = os.path.join(models_dir, "text_generator.keras")
    vocab_path = os.path.join(models_dir, "vocabulary.json")
    config_path = os.path.join(models_dir, "model_config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    model = keras.models.load_model(model_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_payload = json.load(f)
    char_to_idx = vocab_payload["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab_payload["idx_to_char"].items()}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return model, char_to_idx, idx_to_char, config


def sample_next_char(preds: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample next character index from probability distribution using temperature.
    
    Formula:
      p_i = exp(log(p_i) / T) / sum(exp(log(p_j) / T))
      
    Low T (< 0.5): Conservative, deterministic, sharp distribution.
    High T (> 1.0): Varied, diverse, flattened distribution.
    """
    preds = np.asarray(preds, dtype=np.float64)
    # Clip to avoid log(0)
    preds = np.clip(preds, 1e-12, 1.0)

    # Validate temperature
    if temperature <= 0.01:
        # Near zero behaves as argmax (greedy)
        return int(np.argmax(preds))

    # Log probabilities with temperature scaling
    log_preds = np.log(preds) / float(temperature)
    # Numerical stability shift
    log_preds = log_preds - np.max(log_preds)
    exp_preds = np.exp(log_preds)
    probas = exp_preds / np.sum(exp_preds)

    # Re-normalize in case of float rounding
    probas = probas / np.sum(probas)

    # Multinomial sampling
    sampled_indices = np.random.multinomial(1, probas, 1)
    return int(np.argmax(sampled_indices))


def encode_seed_text(
    seed_text: str,
    char_to_idx: Dict[str, int],
    seq_length: int,
    unknown_char_strategy: str = "ignore",
) -> List[int]:
    """
    Encode seed string into integer indices of fixed length `seq_length`.
    
    Strategies for unknown characters:
    - 'ignore': Discard characters not present in vocabulary.
    - 'space': Map unknown character to space (' ') if present, else discard.
    """
    encoded: List[int] = []
    for ch in seed_text:
        if ch in char_to_idx:
            encoded.append(char_to_idx[ch])
        elif unknown_char_strategy == "space" and " " in char_to_idx:
            encoded.append(char_to_idx[" "])
        # else 'ignore' drops the character

    if not encoded:
        # Fallback if seed was completely empty or completely unknown chars
        fallback_char = " " if " " in char_to_idx else list(char_to_idx.keys())[0]
        encoded = [char_to_idx[fallback_char]]

    # Pad or truncate to seq_length
    if len(encoded) < seq_length:
        pad_char = " " if " " in char_to_idx else list(char_to_idx.keys())[0]
        pad_idx = char_to_idx[pad_char]
        encoded = [pad_idx] * (seq_length - len(encoded)) + encoded
    else:
        encoded = encoded[-seq_length:]

    return encoded


def generate_text(
    model: keras.Model,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    seq_length: int,
    seed_text: str,
    gen_length: int = 200,
    temperature: float = 0.8,
) -> str:
    """
    Autoregressively generate text starting from a seed prompt.
    Returns the generated new characters.
    """
    if gen_length <= 0:
        return ""

    current_sequence = encode_seed_text(seed_text, char_to_idx, seq_length)
    generated_chars: List[str] = []

    for _ in range(gen_length):
        x_input = np.array([current_sequence], dtype=np.int32)
        preds = model.predict(x_input, verbose=0)[0]
        next_idx = sample_next_char(preds, temperature=temperature)
        next_char = idx_to_char.get(next_idx, "?")

        generated_chars.append(next_char)
        # Advance sequence window
        current_sequence = current_sequence[1:] + [next_idx]

    return "".join(generated_chars)
