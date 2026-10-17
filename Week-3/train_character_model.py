"""
train_character_model.py
========================
Script to train the character-level LSTM on TinyShakespeare and save artifacts
to Week-3/models/ for reuse across 02_character_level_text_generation.ipynb,
05_lstm_text_generation_app.ipynb, and the Streamlit application.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add Week-3 to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import (
    build_char_lstm_model,
    build_vocabulary,
    save_artifacts,
    generate_text,
)

def main():
    print("TensorFlow Version:", tf.__version__)
    data_path = os.path.join(os.path.dirname(__file__), "data", "shakespeare.txt")
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Corpus not found at {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Use first 120,000 characters for fast yet high-quality CPU training
    training_text = full_text[:120000]
    print(f"Total corpus length: {len(full_text):,} characters")
    print(f"Training slice length: {len(training_text):,} characters")

    unique_chars, char_to_idx, idx_to_char = build_vocabulary(training_text)
    vocab_size = len(unique_chars)
    print(f"Vocabulary size: {vocab_size} unique characters")

    # Hyperparameters
    seq_length = 40
    step = 3
    embedding_dim = 64
    lstm_units = 128
    batch_size = 128
    epochs = 18

    # Create sequences
    sentences = []
    next_chars = []
    for i in range(0, len(training_text) - seq_length, step):
        sentences.append(training_text[i : i + seq_length])
        next_chars.append(training_text[i + seq_length])

    num_samples = len(sentences)
    print(f"Number of training sequences: {num_samples:,}")

    # Vectorize
    X = np.zeros((num_samples, seq_length), dtype=np.int32)
    y = np.zeros((num_samples,), dtype=np.int32)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t] = char_to_idx[char]
        y[i] = char_to_idx[next_chars[i]]

    print("Building model...")
    model = build_char_lstm_model(
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout_rate=0.2,
        learning_rate=0.005,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=2,
            min_lr=1e-4,
            verbose=1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        X,
        y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    print(f"Final training loss: {final_loss:.4f}, accuracy: {final_acc:.4f}")

    # Save artifacts
    print(f"Saving artifacts to {models_dir}...")
    save_artifacts(
        models_dir=models_dir,
        model=model,
        unique_chars=unique_chars,
        seq_length=seq_length,
        additional_config={
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "final_loss": round(float(final_loss), 4),
            "final_accuracy": round(float(final_acc), 4),
            "epochs_trained": len(history.history["loss"]),
            "corpus": "TinyShakespeare (first 120k chars)",
        },
    )
    print("Artifacts successfully saved!")

    # Test generation with different temperatures
    seed = "To be, or not to be, that is the question:"
    print(f"\nSeed: '{seed}'")
    for temp in [0.4, 0.8, 1.2]:
        generated = generate_text(
            model=model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            seq_length=seq_length,
            seed_text=seed,
            gen_length=150,
            temperature=temp,
        )
        print(f"\n--- Temperature {temp} ---\n{seed}{generated}\n")

if __name__ == "__main__":
    main()
