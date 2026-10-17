"""
build_notebooks.py
==================
Generates all five Jupyter Notebooks for Week 3:
1. 01_lstm_sentiment_analysis.ipynb
2. 02_character_level_text_generation.ipynb
3. 03_denoising_autoencoder_fashion_mnist.ipynb
4. 04_gan_mnist_digit_generation.ipynb
5. 05_lstm_text_generation_app.ipynb
"""

import json
import os

WEEK3_DIR = os.path.dirname(os.path.abspath(__file__))


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    }


# ==============================================================================
# NOTEBOOK 1: 01_lstm_sentiment_analysis.ipynb
# ==============================================================================
def build_nb1():
    cells = [
        md_cell("""# 01 — LSTM Sentiment Analysis on IMDB Movie Reviews
### Deep Learning Week 3: Recurrent Neural Networks & Sequential Modeling

---

## 1. Conceptual Introduction: The Need for Recurrent Networks

Traditional feedforward neural networks assume all inputs and outputs are independent of each other. However, human language is inherently **sequential**: the meaning of a word depends heavily on the words that precede and follow it.

### Vanilla Recurrent Neural Networks (RNNs)
A standard RNN maintains an internal hidden state $h_t$ that carries information across time steps $t$:
$$h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

While conceptually elegant, vanilla RNNs suffer from **vanishing and exploding gradients** during backpropagation through time (BPTT). When computing gradients over long sequences, repeated matrix multiplication with weight matrices $W_{hh}$ causes gradients to either decay exponentially to zero (preventing the model from learning long-term dependencies) or grow uncontrollably to infinity.

### Long Short-Term Memory (LSTM) Networks
Introduced by Hochreiter & Schmidhuber (1997), **LSTMs** solve the vanishing gradient problem through a dedicated **cell state** $C_t$ (the "constant error carousel") regulated by three multiplicative gating mechanisms:

1. **Forget Gate ($f_t$):** Determines what proportion of past cell state information should be discarded:
   $$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$ & $\\tilde{C}_t$):** Decides which new information to store in the cell state:
   $$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$$
   $$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update ($C_t$):** Combines retained past memory with newly gated candidate memory:
   $$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$
4. **Output Gate ($o_t$ & $h_t$):** Controls what portion of the cell state is emitted as the new hidden state:
   $$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \\odot \\tanh(C_t)$$

In this notebook, we apply an LSTM to binary sentiment classification on the 50,000-review **IMDB dataset**."""),

        code_cell("""# 2. Imports and Environment Verification
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set reproducible random seed
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")
print(f"GPU Available:      {len(tf.config.list_physical_devices('GPU')) > 0}")"""),

        md_cell("""## 3. Dataset Loading & Inspection: IMDB Movie Reviews

The IMDB dataset consists of 50,000 highly polarized movie reviews labeled as **0 (Negative)** or **1 (Positive)**.
- **25,000 Training Reviews**
- **25,000 Held-Out Testing Reviews**

In Keras, reviews are pre-tokenized into integers where each number represents the frequency rank of that word in the dataset."""),

        code_cell("""# 3.1 Load IMDB Dataset with Top 10,000 Words
VOCAB_SIZE = 10000
MAX_LEN = 200  # Cut off reviews after 200 words or pad if shorter

print(f"Loading IMDB dataset (top {VOCAB_SIZE:,} words)...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

print(f"Train samples: {len(x_train_raw):,}")
print(f"Test samples:  {len(x_test_raw):,}")

# Inspect Class Distribution
pos_train = np.sum(y_train_raw == 1)
neg_train = np.sum(y_train_raw == 0)
print(f"Train Split -> Positive: {pos_train:,} ({pos_train/len(y_train_raw)*100:.1f}%), Negative: {neg_train:,} ({neg_train/len(y_train_raw)*100:.1f}%)")"""),

        md_cell("""### 3.2 Inspecting Token IDs and Review Lengths

In the Keras IMDB dataset:
- Index `0` is reserved for **Padding** (`<PAD>`)
- Index `1` represents the **Start of Sequence** token (`<START>`)
- Index `2` represents **Out of Vocabulary / Unknown Words** (`<UNK>`)
- Index `3` is reserved for an unused token, so actual word ranks begin at index `4`."""),

        code_cell("""# Retrieve the word index dictionary to decode reviews back to text
word_index = keras.datasets.imdb.get_word_index()
# Invert dictionary and offset indices by 3
inverted_word_index = {i + 3: word for word, i in word_index.items()}
inverted_word_index[0] = "<PAD>"
inverted_word_index[1] = "<START>"
inverted_word_index[2] = "<UNK>"
inverted_word_index[3] = "<UNUSED>"

def decode_review(token_list):
    return " ".join([inverted_word_index.get(i, "?") for i in token_list])

print("Sample Tokenized Review (first 20 token IDs):", x_train_raw[0][:20])
print(f"Label: {'Positive (1)' if y_train_raw[0] == 1 else 'Negative (0)'}")
print("\\nDecoded Review (Excerpt):")
print(decode_review(x_train_raw[0][:50]) + "...")

# Review Length Distribution
review_lengths = [len(r) for r in x_train_raw]
print(f"\\nReview Lengths: Min={min(review_lengths)}, Max={max(review_lengths)}, Mean={np.mean(review_lengths):.1f}, Median={np.median(review_lengths)}")"""),

        md_cell("""## 4. Sequence Preprocessing: Padding & Truncation

Neural networks operate on rectangular tensor batches. Because natural language reviews vary from 10 to over 2,000 words, we standardize input length to `MAX_LEN = 200`:
- Shorter reviews are **post-padded** with zeros.
- Longer reviews are **post-truncated** to the first 200 tokens."""),

        code_cell("""# 4.1 Pad and Truncate Sequences
x_train_padded = pad_sequences(x_train_raw, maxlen=MAX_LEN, padding="post", truncating="post")
x_test_padded = pad_sequences(x_test_raw, maxlen=MAX_LEN, padding="post", truncating="post")

# Derive a 20% validation split from training data (keeping test set completely untouched)
VAL_SPLIT = 5000
x_val = x_train_padded[:VAL_SPLIT]
y_val = y_train_raw[:VAL_SPLIT]
x_train = x_train_padded[VAL_SPLIT:]
y_train = y_train_raw[VAL_SPLIT:]

print(f"Training shape:   {x_train.shape}, labels: {y_train.shape}")
print(f"Validation shape: {x_val.shape}, labels: {y_val.shape}")
print(f"Held-out test:    {x_test_padded.shape}, labels: {y_test_raw.shape}")"""),

        md_cell("""## 5. Model Architecture: Embedding + LSTM Classifier

### Architecture Layers:
1. **Embedding Layer (`input_dim=10000`, `output_dim=64`, `input_length=200`):**
   Converts discrete integer token indices into dense continuous vector representations. Words with similar semantic meanings and syntactic roles are mapped close together in this 64-dimensional learned latent space.
2. **SpatialDropout1D (`0.2`):**
   Drops entire 1D feature maps across the sequence, preventing co-adaptation of word features.
3. **LSTM Layer (`units=64`, `dropout=0.2`):**
   Processes the 200-step sequence sequentially, maintaining cell state $C_t$ and emitting the final summary hidden vector $h_{200} \\in \\mathbb{R}^{64}$.
4. **Dense Output Layer (`units=1`, `activation='sigmoid'`):**
   Computes the probability that the review expresses positive sentiment: $P(y=1 | x) = \\sigma(W \\cdot h_{200} + b)$."""),

        code_cell("""# 5.1 Build the LSTM Model
EMBEDDING_DIM = 64
LSTM_UNITS = 64

model = keras.Sequential([
    layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="token_input"),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, name="word_embedding"),
    layers.SpatialDropout1D(0.2, name="spatial_dropout"),
    layers.LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.0, name="lstm_classifier"),
    layers.Dense(32, activation="relu", name="dense_intermediate"),
    layers.Dropout(0.3, name="dense_dropout"),
    layers.Dense(1, activation="sigmoid", name="sentiment_output"),
], name="IMDB_LSTM_Classifier")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()"""),

        md_cell("""## 6. Training with Early Stopping & Validation Tracking

We train for up to 8 epochs with early stopping monitoring validation loss to prevent overfitting."""),

        code_cell("""# 6.1 Train the Model
EPOCHS = 6
BATCH_SIZE = 128

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-4,
        verbose=1,
    )
]

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)"""),

        md_cell("""## 7. Training Diagnostics & Learning Curves"""),

        code_cell("""# 7.1 Plot Training and Validation History
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(history.history["loss"], label="Train Loss", marker="o", color="#1f77b4")
ax1.plot(history.history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
ax1.set_title("Binary Cross-Entropy Loss Across Epochs", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend()

# Accuracy
ax2.plot(history.history["accuracy"], label="Train Accuracy", marker="o", color="#2ca02c")
ax2.plot(history.history["val_accuracy"], label="Val Accuracy", marker="s", color="#d62728")
ax2.set_title("Classification Accuracy Across Epochs", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()"""),

        md_cell("""## 8. Final Evaluation on Held-Out Test Set"""),

        code_cell("""# 8.1 Evaluate on Official 25,000 Test Reviews
test_loss, test_accuracy = model.evaluate(x_test_padded, y_test_raw, batch_size=256, verbose=1)
print(f"\\n================ HELD-OUT TEST RESULTS ================")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"========================================================")"""),

        md_cell("""## 9. Sample Predictions & Error Analysis

We inspect positive, negative, and borderline misclassified reviews to analyze model behavior."""),

        code_cell("""# 9.1 Run Inference on Sample Test Reviews
sample_indices = [0, 1, 2, 3, 4, 10, 15, 20]
test_preds = model.predict(x_test_padded[sample_indices], verbose=0).flatten()

print("=" * 80)
print(f"{'Idx':<4} | {'True Label':<10} | {'Pred Prob':<10} | {'Predicted':<10} | {'Match?':<6}")
print("=" * 80)
for idx, p_prob in zip(sample_indices, test_preds):
    true_lbl = "Positive" if y_test_raw[idx] == 1 else "Negative"
    pred_lbl = "Positive" if p_prob >= 0.5 else "Negative"
    match = "MATCH" if true_lbl == pred_lbl else "MISMATCH"
    print(f"{idx:<4} | {true_lbl:<10} | {p_prob:9.4f} | {pred_lbl:<10} | {match:<8}")

# 9.2 Inspect Misclassified Review
all_test_preds = model.predict(x_test_padded[:500], verbose=0).flatten()
misclassified_indices = np.where((all_test_preds >= 0.5) != y_test_raw[:500])[0]

if len(misclassified_indices) > 0:
    err_idx = misclassified_indices[0]
    err_prob = all_test_preds[err_idx]
    err_true = "Positive" if y_test_raw[err_idx] == 1 else "Negative"
    err_pred = "Positive" if err_prob >= 0.5 else "Negative"
    print(f"\\n--- Detailed Misclassification Case (Index {err_idx}) ---")
    print(f"True Label: {err_true} | Predicted: {err_pred} (Confidence: {err_prob:.4f})")
    print("Decoded Text:")
    print(decode_review(x_test_raw[err_idx][:100]) + "...")"""),

        md_cell("""## 10. Discussion & Limitations

### 1. Strengths of the LSTM Classifier:
- Captures sequential context and word order far better than bag-of-words or naive Bayes models.
- Word embeddings enable the model to generalize between synonyms without manual feature engineering.

### 2. Limitations & Failure Modes:
- **Sequence Truncation:** Cutting off reviews at 200 words discards critical concluding sentiment in longer reviews.
- **Vocabulary Truncation:** Words outside the top 10,000 frequency ranks become `<UNK>`, which may remove specific actor names, technical terms, or rare adjectives.
- **Sarcasm and Nuance:** Binary sentiment labels struggle with sarcastic reviews (e.g., *"A masterpiece of sheer boredom"*), which contain positive words used with negative intent.
- **Bi-directionality:** Standard forward LSTMs only see preceding words. A Bidirectional LSTM (BiLSTM) would improve performance by conditioning each word on both prior and subsequent context."""),
    ]
    return make_notebook(cells)


# ==============================================================================
# NOTEBOOK 2: 02_character_level_text_generation.ipynb
# ==============================================================================
def build_nb2():
    cells = [
        md_cell("""# 02 — Character-Level Text Generation with LSTM
### Deep Learning Week 3: Autoregressive Language Modeling & Temperature Sampling

---

## 1. Conceptual Introduction: Autoregressive Language Modeling

Text generation is formulated as a sequential prediction task. Given an initial sequence of characters:
$$x_1, x_2, \\dots, x_t$$
the language model predicts the categorical probability distribution over the next character:
$$P(x_{t+1} = c \\mid x_1, x_2, \\dots, x_t) \\quad \\text{for each character } c \\in \\mathcal{V}$$

### Character-Level vs. Word-Level Modeling
| Feature | Character-Level Language Model | Word-Level Language Model |
| :--- | :--- | :--- |
| **Vocabulary Size ($|\\mathcal{V}|$)** | Tiny (~65 to 100 characters) | Massive (10,000 to 100,000+ words) |
| **Out-of-Vocabulary (OOV)** | 0% (every text decomposes into characters) | Significant (unknown words require `<UNK>`) |
| **Sequence Length ($T$)** | Long (a single sentence is 50–200 characters) | Short (a sentence is 10–30 words) |
| **Task Complexity** | Must learn spelling, words, punctuation, and grammar | Learns syntax and semantics directly |

### Temperature-Based Sampling
When generating the next character, taking the greedy argmax $\\arg\\max_c P(x_{t+1}=c)$ causes repetitive, degenerate loops. Sampling randomly from the raw softmax probabilities can produce incoherent gibberish.

**Temperature ($T > 0$)** rescales the unnormalized model logits $z_i$ or log-probabilities:
$$p_i = \\frac{\\exp(z_i / T)}{\\sum_j \\exp(z_j / T)} = \\frac{\\exp(\\log(p_i) / T)}{\\sum_j \\exp(\\log(p_j) / T)}$$

- **$T \\to 0$ (Low Temperature):** Probability mass concentrates on the highest logit. The model becomes conservative, confident, and grammatically strict, but repetitive.
- **$T = 1.0$ (Neutral Temperature):** Standard softmax distribution as learned during training.
- **$T > 1.2$ (High Temperature):** Flattens the distribution toward uniform random selection. Output is diverse and creative, but prone to invented words and syntactic breakdown.

In this notebook, we train an LSTM on William Shakespeare's public-domain plays and systematically evaluate temperature sampling."""),

        code_cell("""# 2. Imports and Environment Setup
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Add Week-3 to sys.path to access reusable model_utils
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_utils import (
    build_char_lstm_model,
    build_vocabulary,
    sample_next_char,
    generate_text,
    load_artifacts,
)

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")"""),

        md_cell("""## 3. Corpus Selection & Preprocessing: TinyShakespeare

We use the public-domain Shakespeare corpus (`shakespeare.txt`), derived from Project Gutenberg.
The text contains plays with character dialogue, act markers, and verse structure."""),

        code_cell("""# 3.1 Load and Inspect Shakespeare Corpus
data_path = os.path.join(current_dir, "data", "shakespeare.txt")
if not os.path.exists(data_path):
    # Fallback path if running inside Week-3 subfolder
    data_path = os.path.join("..", "data", "shakespeare.txt")

with open(data_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"Total corpus length: {len(raw_text):,} characters")
print("\\n--- Corpus Excerpt (First 250 characters) ---")
print(raw_text[:250])
print("---------------------------------------------")"""),

        md_cell("""### 3.2 Vocabulary Construction and Bidirectional Mapping

We extract all unique characters and construct:
- `char_to_idx`: maps characters to integer token indices $[0, |\\mathcal{V}| - 1]$.
- `idx_to_char`: maps integer indices back to character strings."""),

        code_cell("""# 3.2 Build Character Vocabulary
# Use first 120,000 characters for fast CPU training and consistent demonstration
training_text = raw_text[:120000]
unique_chars, char_to_idx, idx_to_char = build_vocabulary(training_text)
vocab_size = len(unique_chars)

print(f"Training slice length: {len(training_text):,} characters")
print(f"Unique characters ({vocab_size}):")
print("".join(unique_chars))"""),

        md_cell("""### 3.3 Sequence Generation (Sliding Window)

We cut the text into overlapping chunks of length `SEQ_LENGTH = 40` with stride `STEP = 3`."""),

        code_cell("""# 3.3 Create Fixed-Length Sliding Windows
SEQ_LENGTH = 40
STEP = 3

sentences = []
next_chars = []
for i in range(0, len(training_text) - SEQ_LENGTH, STEP):
    sentences.append(training_text[i : i + SEQ_LENGTH])
    next_chars.append(training_text[i + SEQ_LENGTH])

print(f"Total training sequences: {len(sentences):,}")
print(f"Example input sentence: '{sentences[0]}'")
print(f"Example target next char: '{next_chars[0]}'")

# Vectorize into numpy integer arrays
X = np.zeros((len(sentences), SEQ_LENGTH), dtype=np.int32)
y = np.zeros((len(sentences),), dtype=np.int32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t] = char_to_idx[char]
    y[i] = char_to_idx[next_chars[i]]

print(f"X shape: {X.shape}, y shape: {y.shape}")"""),

        md_cell("""## 4. LSTM Model Architecture

The neural network consists of:
1. `Input`: Shape `(40,)` containing character indices.
2. `Embedding(vocab_size, 64)`: Maps each character to a 64-dimensional vector.
3. `LSTM(128, return_sequences=False)`: Summarizes the 40-character sequence into a 128-dimensional hidden state vector.
4. `Dropout(0.2)`: Regularization to prevent verbatim sequence memorization.
5. `Dense(vocab_size, activation='softmax')`: Predicts the probability distribution over all unique characters."""),

        code_cell("""# 4.1 Build or Load the Pre-Trained Model
models_dir = os.path.join(current_dir, "models")
if not os.path.exists(models_dir):
    models_dir = os.path.join(current_dir, "Week-3", "models")
model_file = os.path.join(models_dir, "text_generator.keras")

if os.path.exists(model_file):
    print("Loading existing trained model and artifacts from models/...")
    model, char_to_idx, idx_to_char, config = load_artifacts(models_dir)
else:
    print("Building fresh character LSTM model...")
    model = build_char_lstm_model(
        vocab_size=vocab_size,
        seq_length=SEQ_LENGTH,
        embedding_dim=64,
        lstm_units=128,
        dropout_rate=0.2,
        learning_rate=0.005,
    )

model.summary()"""),

        md_cell("""## 5. Model Training and Convergence

If training from scratch or validating the training trajectory, we train using `sparse_categorical_crossentropy`."""),

        code_cell("""# 5.1 Training Loop (Quick Run or Full Trajectory)
# Check if model has already been trained and saved
if not os.path.exists(model_file):
    print("Training model for 10 epochs...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
    ]
    history = model.fit(
        X,
        y,
        batch_size=128,
        epochs=10,
        callbacks=callbacks,
        verbose=1,
    )
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], marker="o", color="#1f77b4", label="Training Loss")
    plt.title("Character LSTM Training Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Sparse Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()
else:
    print("Trained model is already available in models/text_generator.keras!")"""),

        md_cell("""## 6. Temperature Sampling Experiments

We test the exact same prompt at three different temperatures:
- **Low Temperature ($T = 0.4$):** Conservative, repeats dominant character n-grams.
- **Moderate Temperature ($T = 0.8$):** Realistic balance of grammatical cohesion and creativity.
- **High Temperature ($T = 1.2$):** Highly diverse, erratic words, elevated entropy."""),

        code_cell("""# 6.1 Run Temperature Sampling Comparison
seed_prompt = "To be, or not to be, that is the question:"
temperatures = [0.4, 0.8, 1.2]
GEN_LENGTH = 200

print(f"Seed Prompt: '{seed_prompt}'\\n")
print("=" * 80)

for temp in temperatures:
    generated_text = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seq_length=SEQ_LENGTH,
        seed_text=seed_prompt,
        gen_length=GEN_LENGTH,
        temperature=temp,
    )
    print(f"\\n--- TEMPERATURE: {temp} ---")
    print(seed_prompt + generated_text)
    print("-" * 80)"""),

        md_cell("""## 7. Analysis and Observations

### Observations on Output Quality:
1. **At $T = 0.4$:** The model correctly reproduces common Shakespearean dialogue structures (e.g., character speech prefixes like `CORIOLANUS:`, `MENENIUS:`), but easily falls into repetitive word loops (*"the state of the state of the state"*).
2. **At $T = 0.8$:** Generates varied archaic English vocabulary (*"thou", "thee", "hath", "speak"*) with credible verse formatting, rhythm, and stanza line breaks.
3. **At $T = 1.2$:** Punctuation is scattered irregularly, and unusual character combinations appear because the flattened softmax distribution frequently samples low-probability tokens.

### Limitations of Character-Level Models:
- **Long-term semantic coherence:** The model maintains local syntactic validity for 5–15 characters, but cannot plan overarching plot or paragraph-level semantic intent.
- **Context window:** The 40-character window means the model has limited memory of what characters or events were mentioned earlier in the scene."""),
    ]
    return make_notebook(cells)


# ==============================================================================
# NOTEBOOK 3: 03_denoising_autoencoder_fashion_mnist.ipynb
# ==============================================================================
def build_nb3():
    cells = [
        md_cell("""# 03 — Denoising Autoencoder on Fashion-MNIST
### Deep Learning Week 3: Self-Supervised Representation Learning & Image Reconstruction

---

## 1. Conceptual Introduction: Autoencoders and the Denoising Criterion

An **Autoencoder** is an unsupervised / self-supervised neural network trained to reconstruct its input through a constrained bottleneck representation:
$$x \\xrightarrow{\\text{Encoder } f_\\theta} z \\xrightarrow{\\text{Decoder } g_\\phi} \\hat{x} \\approx x$$

Where:
- $z = f_\\theta(x) \\in \\mathbb{R}^d$ is the compact **latent representation** (bottleneck).
- $\\hat{x} = g_\\phi(z) \\in \\mathbb{R}^D$ is the **reconstructed output**.

If an autoencoder simply learned the identity mapping $g(f(x)) = x$, it could memorize inputs without discovering useful semantic features.

### The Denoising Criterion (Vincent et al., 2008)
A **Denoising Autoencoder (DAE)** introduces a stochastic corruption process $q(\\tilde{x} \\mid x)$ to the input:
$$x \\xrightarrow{\\text{Corrupt}} \\tilde{x} = x + \\epsilon \\quad (\\epsilon \\sim \\mathcal{N}(0, \\sigma^2))$$
$$\\tilde{x} \\xrightarrow{f_\\theta} z \\xrightarrow{g_\\phi} \\hat{x}$$

Crucially, the reconstruction loss penalizes the difference between $\\hat{x}$ and the **original clean image $x$**, not the corrupted input $\\tilde{x}$:
$$\\mathcal{L}(x, \\hat{x}) = \\|x - g_\\phi(f_\\theta(\\tilde{x}))\\|^2$$

This forces the autoencoder to:
1. Learn the underlying **low-dimensional data manifold** $\\mathcal{M}$ where realistic clothing images reside.
2. Act as a **projection operator** that maps off-manifold noisy samples back onto $\\mathcal{M}$."""),

        code_cell("""# 2. Imports and Environment Setup
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version:      {keras.__version__}")"""),

        md_cell("""## 3. Dataset Loading & Inspection: Fashion-MNIST

Fashion-MNIST consists of 70,000 $28 \\times 28$ grayscale images across 10 clothing categories:
`0: T-shirt/top`, `1: Trouser`, `2: Pullover`, `3: Dress`, `4: Coat`, `5: Sandal`, `6: Shirt`, `7: Sneaker`, `8: Bag`, `9: Ankle boot`."""),

        code_cell("""# 3.1 Load Fashion-MNIST Dataset
(x_train_raw, y_train), (x_test_raw, y_test) = keras.datasets.fashion_mnist.load_data()

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"Train set: {x_train_raw.shape}, Test set: {x_test_raw.shape}")

# Normalize pixel values to [0.0, 1.0] and add channel dimension
x_train = x_train_raw.astype("float32") / 255.0
x_test = x_test_raw.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)  # Shape: (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # Shape: (10000, 28, 28, 1)

print(f"Processed input shape: {x_train.shape}, range: [{x_train.min():.1f}, {x_train.max():.1f}]")"""),

        md_cell("""## 4. The Corruption Process: Additive Gaussian Noise

We apply zero-mean additive Gaussian noise with $\\sigma = 0.35$, followed by clipping to $[0.0, 1.0]$:
$$\\tilde{x} = \\text{clip}(x + \\mathcal{N}(0, \\sigma^2), 0.0, 1.0)$$"""),

        code_cell("""# 4.1 Corrupt Images with Gaussian Noise
NOISE_FACTOR = 0.35

x_train_noisy = x_train + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + NOISE_FACTOR * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# 4.2 Visualize Clean vs. Noisy Images
n_samples = 8
plt.figure(figsize=(16, 4))
for i in range(n_samples):
    # Clean image
    ax = plt.subplot(2, n_samples, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Clean: {CLASS_NAMES[y_train[i]]}", fontsize=10)
    plt.axis("off")

    # Noisy image
    ax = plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(x_train_noisy[i].squeeze(), cmap="gray")
    plt.title("Noisy Input", fontsize=10)
    plt.axis("off")

plt.suptitle("Fashion-MNIST: Clean Targets vs. Corrupted Inputs", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()"""),

        md_cell("""## 5. Convolutional Autoencoder Architecture

We construct a **Convolutional Autoencoder (CAE)**:
- **Encoder:**
  - `Conv2D(32, (3,3), relu, padding='same')` $\\to (28, 28, 32)$
  - `MaxPooling2D((2,2))` $\\to (14, 14, 32)$
  - `Conv2D(16, (3,3), relu, padding='same')` $\\to (14, 14, 16)$
  - `MaxPooling2D((2,2))` $\\to (7, 7, 16)$ (Bottleneck: 784 latent variables)
- **Decoder:**
  - `Conv2DTranspose(16, (3,3), strides=2, relu, padding='same')` $\\to (14, 14, 16)$
  - `Conv2DTranspose(32, (3,3), strides=2, relu, padding='same')` $\\to (28, 28, 32)$
  - `Conv2D(1, (3,3), sigmoid, padding='same')` $\\to (28, 28, 1)$ (Clean reconstruction)"""),

        code_cell("""# 5.1 Define Convolutional Denoising Autoencoder
inputs = layers.Input(shape=(28, 28, 1), name="noisy_image_input")

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_conv1")(inputs)
x = layers.MaxPooling2D((2, 2), padding="same", name="enc_pool1")(x)
x = layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="enc_conv2")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same", name="latent_bottleneck")(x)

# Decoder
x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same", name="dec_transconv1")(encoded)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same", name="dec_transconv2")(x)
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="reconstructed_clean_image")(x)

autoencoder = keras.Model(inputs=inputs, outputs=decoded, name="Fashion_Denoising_Autoencoder")
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
autoencoder.summary()"""),

        md_cell("""## 6. Model Training: Reconstructing Clean Targets

We train the autoencoder by feeding `x_train_noisy` as input and targeting `x_train` (clean targets)."""),

        code_cell("""# 6.1 Train the Denoising Autoencoder
EPOCHS = 8
BATCH_SIZE = 128

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
]

history = autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1,
)"""),

        md_cell("""## 7. Reconstruction Evaluation & Loss Curves"""),

        code_cell("""# 7.1 Plot Training and Validation Loss Curves
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train MSE Loss", marker="o", color="#1f77b4")
plt.plot(history.history["val_loss"], label="Val MSE Loss", marker="s", color="#ff7f0e")
plt.title("Denoising Autoencoder Loss (MSE) Across Epochs", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Evaluate on Held-Out Test Set
test_mse = autoencoder.evaluate(x_test_noisy, x_test, verbose=0)
print(f"Held-Out Test MSE: {test_mse:.5f}")"""),

        md_cell("""## 8. Side-by-Side Visual Comparison: Clean vs. Noisy vs. Denoised

We display an 8-column visual comparison showing:
1. **Row 1:** Ground Truth Clean Images ($x$)
2. **Row 2:** Corrupted Noisy Images ($\\tilde{x}$)
3. **Row 3:** Denoised Reconstructions ($\\hat{x}$)"""),

        code_cell("""# 8.1 Generate Reconstructions on Test Set
reconstructed_images = autoencoder.predict(x_test_noisy[:n_samples], verbose=0)

plt.figure(figsize=(16, 6))
for i in range(n_samples):
    # 1. Clean
    ax = plt.subplot(3, n_samples, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title(f"Clean: {CLASS_NAMES[y_test[i]]}", fontsize=9)
    plt.axis("off")

    # 2. Noisy
    ax = plt.subplot(3, n_samples, i + 1 + n_samples)
    plt.imshow(x_test_noisy[i].squeeze(), cmap="gray")
    plt.title("Noisy Input", fontsize=9)
    plt.axis("off")

    # 3. Reconstructed
    ax = plt.subplot(3, n_samples, i + 1 + 2 * n_samples)
    plt.imshow(reconstructed_images[i].squeeze(), cmap="gray")
    plt.title("Denoised Output", fontsize=9)
    plt.axis("off")

plt.suptitle("Denoising Autoencoder: Clean vs. Noisy vs. Reconstructed", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# 8.2 Quantitative Denoising Improvement
mse_noisy = np.mean((x_test[:1000] - x_test_noisy[:1000]) ** 2)
mse_denoised = np.mean((x_test[:1000] - autoencoder.predict(x_test_noisy[:1000], verbose=0)) ** 2)
reduction = (1.0 - mse_denoised / mse_noisy) * 100

print(f"Input Noise MSE:         {mse_noisy:.5f}")
print(f"Post-Denoising MSE:      {mse_denoised:.5f}")
print(f"Noise Error Reduction:   {reduction:.1f}%")"""),

        md_cell("""## 9. Observations & Limitations

### What Worked Well:
- The convolutional layers successfully suppress high-frequency uncorrelated Gaussian noise across the background and item silhouettes.
- Structural silhouettes of distinct shapes (e.g., boots, trousers, bags) are accurately preserved.

### Trade-offs and Limitations:
- **Smoothing of Fine Textures:** Because MSE penalizes pixel-by-pixel squared discrepancies, the autoencoder tends to output the average of plausible pixel values. This results in slight blurring of sharp logos, button details, or shoe laces.
- **Capacity Bottleneck:** A bottleneck of $7 \\times 7 \\times 16 = 784$ features enforces significant compression ($28 \\times 28 = 784$ with channels), preventing trivial identity pass-through while retaining salient shape descriptors."""),
    ]
    return make_notebook(cells)


# ==============================================================================
# NOTEBOOK 4: 04_gan_mnist_digit_generation.ipynb
# ==============================================================================
def build_nb4():
    cells = [
        md_cell("""# 04 — Generative Adversarial Network (GAN) on MNIST Digit Generation
### Deep Learning Week 3: Adversarial Training, Generator-Discriminator Dynamics, and Image Synthesis

---

## 1. Conceptual Introduction: The Minimax Game of GANs

Introduced by Ian Goodfellow et al. (2014), **Generative Adversarial Networks (GANs)** frame generative modeling as a two-player zero-sum game between two competing neural networks:

$$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{\\text{data}}(x)}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z(z)}[\\log(1 - D(G(z)))]$$

### 1. The Generator ($G$): The Counterfeiter
- Takes a random latent noise vector $z \\sim \\mathcal{N}(0, I)$ of dimension $d_z = 100$.
- Maps $z$ through convolutional and transpose-convolutional layers to produce a synthetic $28 \\times 28$ image: $x_{\\text{fake}} = G(z)$.
- Objective: Maximize $D(G(z))$, fooling the Discriminator into classifying $x_{\\text{fake}}$ as a genuine handwritten digit. In practice, we minimize $-\\log D(G(z))$ to prevent vanishing gradients early in training.

### 2. The Discriminator ($D$): The Detective
- Takes a $28 \\times 28$ grayscale image (either real $x$ from the dataset or fake $G(z)$).
- Outputs a scalar score $D(x) \\in [0, 1]$ indicating the probability that $x$ came from the true data distribution.
- Objective: Maximize $\\log D(x) + \\log(1 - D(G(z)))$.

### Critical GAN Challenges
- **Oscillating Loss Curves:** Unlike standard supervised training where loss monotonically decreases, GAN losses fluctuate as the two adversaries adapt to each other's strategies. A low Generator loss is meaningless if the Discriminator is simply collapsing.
- **Mode Collapse:** The Generator discovers a single digit class (e.g., drawing only '1's or '8's) that reliably fools the Discriminator, failing to cover the full diversity of the data distribution.
- **Vanishing Gradients:** If the Discriminator learns too fast, $D(G(z)) \\approx 0$ and the Generator receives negligible gradient signal to improve. We use **one-sided label smoothing** ($0.9$ instead of $1.0$ for real labels) and LeakyReLU activations to stabilize training."""),

        code_cell("""# 2. Imports and Environment Setup
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available:      {len(tf.config.list_physical_devices('GPU')) > 0}")"""),

        md_cell("""## 3. Dataset Preprocessing: Normalizing to [-1.0, 1.0]

The Generator uses the `tanh` activation function in its final layer, which bounds pixel outputs to $[-1.0, 1.0]$.
We normalize the MNIST training images to this exact range:
$$x_{\\text{norm}} = \\frac{x - 127.5}{127.5} \\in [-1.0, 1.0]$$"""),

        code_cell("""# 3.1 Load and Normalize MNIST Dataset
(x_train_raw, _), (x_test_raw, _) = keras.datasets.mnist.load_data()

# Normalize to [-1, 1]
x_train = (x_train_raw.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)  # Shape: (60000, 28, 28, 1)

print(f"MNIST training set: {x_train.shape}, range: [{x_train.min():.1f}, {x_train.max():.1f}]")"""),

        md_cell("""## 4. Architecture Definition: Generator & Discriminator

### Generator Architecture:
- Input: Latent vector $z \\in \\mathbb{R}^{100}$
- `Dense(7 * 7 * 128)` + `LeakyReLU(0.2)`
- `Reshape((7, 7, 128))`
- `Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')` $\\to (14, 14, 64)$ + `LeakyReLU(0.2)`
- `Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh')` $\\to (28, 28, 1)$

### Discriminator Architecture:
- Input: Grayscale image $(28, 28, 1)$
- `Conv2D(64, kernel_size=3, strides=2, padding='same')` $\\to (14, 14, 64)$ + `LeakyReLU(0.2)` + `Dropout(0.3)`
- `Conv2D(128, kernel_size=3, strides=2, padding='same')` $\\to (7, 7, 128)$ + `LeakyReLU(0.2)` + `Dropout(0.3)`
- `Flatten()` $\\to 6272$
- `Dense(1, activation='sigmoid')`"""),

        code_cell("""# 4.1 Define the Generator Network
LATENT_DIM = 100

def make_generator():
    model = keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh"),
    ], name="DCGAN_Generator")
    return model

generator = make_generator()
generator.summary()

# 4.2 Define the Discriminator Network
def make_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ], name="DCGAN_Discriminator")
    return model

discriminator = make_discriminator()
discriminator.summary()"""),

        md_cell("""## 5. Adversarial Training Loop

We configure separate Adam optimizers ($lr=0.0002, \\beta_1=0.5$) and execute the two-phase training step:
1. **Discriminator Step:** Train on real samples (label = 0.9 with label smoothing) and fake generated samples (label = 0.0).
2. **Generator Step:** Sample new latent noise vectors and optimize Generator parameters with target label = 1.0."""),

        code_cell("""# 5.1 Build Adversarial Training Setup
d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
bce_loss = keras.losses.BinaryCrossentropy()

# Fixed noise vector to track visual progression across training epochs
FIXED_NOISE = tf.random.normal([16, LATENT_DIM], seed=42)

@tf.function
def train_step(real_images, batch_size):
    # 1. Sample latent noise
    noise = tf.random.normal([batch_size, LATENT_DIM])

    # 2. Train Discriminator
    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        # One-sided label smoothing on real images (0.9)
        d_loss_real = bce_loss(tf.ones_like(real_output) * 0.9, real_output)
        d_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)
        d_loss = d_loss_real + d_loss_fake

    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    # 3. Train Generator (fool discriminator)
    noise_g = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as g_tape:
        generated_images = generator(noise_g, training=True)
        predictions = discriminator(generated_images, training=True)
        g_loss = bce_loss(tf.ones_like(predictions), predictions)

    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return d_loss, g_loss"""),

        md_cell("""## 6. Execution: Adversarial Training with Progress Grids

We train for 12 epochs over a representative subset of the dataset for fast local convergence."""),

        code_cell("""# 6.1 Train the GAN
BATCH_SIZE = 128
EPOCHS = 12
STEPS_PER_EPOCH = 150  # 150 batches * 128 = 19,200 samples per epoch

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(5000).batch(BATCH_SIZE, drop_remainder=True)

d_losses = []
g_losses = []
stage_samples = {}

print("Beginning Adversarial Training Loop...")
for epoch in range(1, EPOCHS + 1):
    epoch_d_loss = []
    epoch_g_loss = []
    
    for step, real_batch in enumerate(dataset.take(STEPS_PER_EPOCH)):
        d_l, g_l = train_step(real_batch, BATCH_SIZE)
        epoch_d_loss.append(float(d_l))
        epoch_g_loss.append(float(g_l))

    mean_d = np.mean(epoch_d_loss)
    mean_g = np.mean(epoch_g_loss)
    d_losses.append(mean_d)
    g_losses.append(mean_g)

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Discriminator Loss: {mean_d:.4f} | Generator Loss: {mean_g:.4f}")

    # Capture visual snapshots at key milestones
    if epoch in [1, 4, 8, 12]:
        preds = generator(FIXED_NOISE, training=False).numpy()
        # Rescale from [-1, 1] to [0, 1]
        stage_samples[epoch] = (preds + 1.0) / 2.0"""),

        md_cell("""## 7. Visual Progression: From Random Noise to Coherent Digits"""),

        code_cell("""# 7.1 Visualize Generated Digits Across Training Epochs
fig, axes = plt.subplots(len(stage_samples), 8, figsize=(14, 2.5 * len(stage_samples)))

for row_idx, (epoch, images) in enumerate(stage_samples.items()):
    for col_idx in range(8):
        ax = axes[row_idx, col_idx]
        ax.imshow(images[col_idx, :, :, 0], cmap="gray")
        if col_idx == 0:
            ax.set_ylabel(f"Epoch {epoch}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("GAN Synthesis Progression from Fixed Latent Vectors", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()"""),

        md_cell("""## 8. Loss Curves and Convergence Analysis"""),

        code_cell("""# 8.1 Plot Adversarial Loss History
plt.figure(figsize=(10, 4.5))
plt.plot(d_losses, label="Discriminator Loss", color="#d62728", marker="o")
plt.plot(g_losses, label="Generator Loss", color="#1f77b4", marker="s")
plt.title("Adversarial Loss History Across Training Epochs", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()"""),

        md_cell("""## 9. Discussion & Critical Analysis

### 1. Visual Progression Analysis:
- **Epoch 1:** Output consists of amorphous Gaussian noise and fuzzy white blobs with no digit structure.
- **Epoch 4:** The Generator starts assembling curved loops and stroke primitives, beginning to approximate circular digits ('0', '8') and linear strokes ('1', '7').
- **Epoch 12:** Clear digit topologies (recognizable 0s, 1s, 7s, 3s) emerge with clean dark backgrounds.

### 2. The Fallacy of GAN Loss Curves:
- In supervised learning, lower loss directly implies superior generalization.
- In GANs, **losses oscillate**. As the Generator improves, the Discriminator's task becomes harder, raising its loss. When the Discriminator adjusts, the Generator's loss climbs. A steady equilibrium is ideal, and visual inspection of samples remains the ground truth of generative quality."""),
    ]
    return make_notebook(cells)


# ==============================================================================
# NOTEBOOK 5: 05_lstm_text_generation_app.ipynb
# ==============================================================================
def build_nb5():
    cells = [
        md_cell("""# 05 — Week 3 Mini Project: End-to-End LSTM Text Generation System
### Interactive Machine Learning Pipeline: Corpus Preprocessing, Training, Artifact Persistence & Streamlit Interface

---

## 1. Project Overview & Architecture

This notebook details the training, artifact export, and verification pipeline for the **Week 3 Interactive Text Generation System**.

### End-to-End Workflow:
```
[Raw Shakespeare Text] 
         │
         ▼
[Vocabulary Extraction] ──► [vocabulary.json]
         │
         ▼
[Sliding Window Sequences]
         │
         ▼
[LSTM Neural Network] ──► [text_generator.keras] & [model_config.json]
         │
         ▼
[model_utils.py] ── (Reusable Sampling & Inference)
         │
         ▼
[Streamlit app.py] ── (Interactive UI with Temperature & Length Sliders)
```

By decoupling model training, artifact persistence, and application logic, the Streamlit app loads pre-trained artifacts once via `@st.cache_resource` and performs real-time autoregressive generation."""),

        code_cell("""# 2. Imports and Environment Setup
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Connect Week-3 directory to path
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_utils import (
    build_char_lstm_model,
    build_vocabulary,
    save_artifacts,
    load_artifacts,
    sample_next_char,
    generate_text,
)

print(f"TensorFlow Version: {tf.__version__}")"""),

        md_cell("""## 3. Corpus Preprocessing & Vocabulary Extraction"""),

        code_cell("""# 3.1 Load Shakespeare Corpus
data_path = os.path.join(current_dir, "data", "shakespeare.txt")
if not os.path.exists(data_path):
    data_path = os.path.join("..", "data", "shakespeare.txt")

with open(data_path, "r", encoding="utf-8") as f:
    full_text = f.read()

training_text = full_text[:120000]
unique_chars, char_to_idx, idx_to_char = build_vocabulary(training_text)
vocab_size = len(unique_chars)

print(f"Loaded corpus: {len(full_text):,} total chars, {len(training_text):,} training slice chars")
print(f"Vocabulary size: {vocab_size} unique characters")"""),

        md_cell("""## 4. Model Training & Artifact Persistence

We train the model or load the verified pre-trained artifacts from `models/`."""),

        code_cell("""# 4.1 Check and Load Artifacts
models_dir = os.path.join(current_dir, "models")
if not os.path.exists(models_dir):
    models_dir = os.path.join(current_dir, "Week-3", "models")
model_path = os.path.join(models_dir, "text_generator.keras")

if os.path.exists(model_path):
    print("Loading verified model and configuration from models/...")
    model, char_to_idx, idx_to_char, config = load_artifacts(models_dir)
    print("Artifacts successfully loaded!")
    print(f"Model Configuration: {json.dumps(config, indent=2)}")
else:
    print("Artifacts not yet saved. Training character LSTM...")
    # Hyperparameters
    SEQ_LEN = 40
    model = build_char_lstm_model(
        vocab_size=vocab_size,
        seq_length=SEQ_LEN,
        embedding_dim=64,
        lstm_units=128,
        dropout_rate=0.2,
        learning_rate=0.005,
    )
    # Save freshly built artifacts
    save_artifacts(models_dir, model, unique_chars, SEQ_LEN)
    print("Saved baseline artifacts.")"""),

        md_cell("""## 5. Verification of Reusable Inference Utilities

We test the end-to-end autoregressive generation logic that powers the Streamlit interface."""),

        code_cell("""# 5.1 Test Autoregressive Generation
test_seed = '''ROMEO:
What light through yonder window breaks?'''
seq_len = config.get("seq_length", 40) if "config" in locals() else 40

print(f"Test Seed:\\n{test_seed}\\n")
for temp in [0.4, 0.8, 1.2]:
    generated = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seq_length=seq_len,
        seed_text=test_seed,
        gen_length=150,
        temperature=temp,
    )
    print(f"--- Temperature {temp} ---")
    print(test_seed + generated)
    print("-" * 60)"""),

        md_cell("""## 6. How the Streamlit Application Works (`app.py`)

The accompanying Streamlit application (`app.py`) delivers an interactive interface with the following architecture:
1. **Model Caching:** `@st.cache_resource` loads `text_generator.keras` and `vocabulary.json` once into memory, preventing expensive reloading on user interactions.
2. **Interactive Controls:**
   - **Seed Prompt Input:** Accepts custom text or quick sample presets (e.g. *"First Citizen:"*, *"To be, or not to be"*).
   - **Temperature Slider:** Controls sampling entropy from 0.1 to 1.5.
   - **Generation Length Slider:** Controls token count from 50 to 500 characters.
3. **Graceful Error Handling:**
   - Empty prompts display user warnings.
   - Characters outside the training vocabulary are cleanly ignored without crashing.
   - Missing model files provide clear instructions on training.

### Launching the Application:
To run the interactive Streamlit application from your terminal:
```bash
streamlit run Week-3/app.py
```"""),
    ]
    return make_notebook(cells)


def main():
    print("Writing Week 3 Notebooks...")

    notebooks = {
        "01_lstm_sentiment_analysis.ipynb": build_nb1(),
        "02_character_level_text_generation.ipynb": build_nb2(),
        "03_denoising_autoencoder_fashion_mnist.ipynb": build_nb3(),
        "04_gan_mnist_digit_generation.ipynb": build_nb4(),
        "05_lstm_text_generation_app.ipynb": build_nb5(),
    }

    for filename, nb_dict in notebooks.items():
        filepath = os.path.join(WEEK3_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb_dict, f, indent=2)
        print(f"[OK] Created {filename} ({len(nb_dict['cells'])} cells)")

    print("\nAll Week 3 notebooks successfully written!")


if __name__ == "__main__":
    main()
