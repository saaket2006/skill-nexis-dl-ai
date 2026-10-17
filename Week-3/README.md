# Week 3 — RNNs, LSTMs, Autoencoders, GANs, and Generative Text Models

This directory contains the complete curriculum, experimental notebooks, interactive Streamlit application, and testing suite for **Week 3** of the Deep Learning & AI training program.

---

## 1. Directory Structure

```
Week-3/
├── 01_lstm_sentiment_analysis.ipynb          # Task A: LSTM sentiment classification on IMDB
├── 02_character_level_text_generation.ipynb   # Task B: Character-level LSTM & Temperature sampling
├── 03_denoising_autoencoder_fashion_mnist.ipynb # Task C: Convolutional Denoising Autoencoder
├── 04_gan_mnist_digit_generation.ipynb       # Task D: Generative Adversarial Network on MNIST
├── 05_lstm_text_generation_app.ipynb         # Mini Project: End-to-end training & export pipeline
├── app.py                                    # Interactive Streamlit Text Generation Web App
├── model_utils.py                            # Reusable inference, vectorization, & sampling module
├── train_character_model.py                  # Standalone training script for character LSTM
├── build_notebooks.py                        # Notebook generation script
├── requirements.txt                          # Scoped dependencies for Week 3
├── README.md                                 # Comprehensive documentation & verified results
├── data/
│   └── shakespeare.txt                       # Public-domain Shakespeare corpus
├── models/
│   ├── .gitkeep
│   ├── text_generator.keras                  # Saved trained character-level LSTM model
│   ├── vocabulary.json                       # Character-to-index and index-to-character maps
│   └── model_config.json                     # Sequence length, embedding dim, units, metadata
└── tests/
    └── test_app.py                           # 10 automated unit & integration tests
```

---

## 2. Notebook Objectives & Conceptual Foundations

### 01. LSTM Sentiment Analysis on IMDB (`01_lstm_sentiment_analysis.ipynb`)
- **Objective:** Train an LSTM binary classifier on 50,000 polarized movie reviews.
- **Concepts:** Vanishing and exploding gradients in recurrent networks; LSTM gating mechanisms ($f_t, i_t, \tilde{C}_t, C_t, o_t, h_t$); dense word embeddings; sequence padding and truncation.
- **Dataset:** Keras IMDB dataset (`tensorflow.keras.datasets.imdb`).
  - 25,000 train reviews, 25,000 held-out test reviews.
  - Vocabulary limit: 10,000 most frequent tokens.
  - Sequence length: 200 tokens (post-padded, post-truncated).
- **Architecture:** `Input(200) -> Embedding(10000, 64) -> SpatialDropout1D(0.2) -> LSTM(64) -> Dense(32, relu) -> Dropout(0.3) -> Dense(1, sigmoid)`.
- **Loss & Optimizer:** Binary cross-entropy with Adam optimizer ($lr=0.001$).
- **Key Results:**
  - Evaluated on held-out 25,000 test reviews.
  - Test Accuracy: ~85.4% – 86.8%.
  - Error analysis demonstrates misclassification on sarcastic reviews and reviews where sentiment shifts only in the untruncated tail.

---

### 02. Character-Level Text Generation with LSTM (`02_character_level_text_generation.ipynb`)
- **Objective:** Train a character-level autoregressive language model on William Shakespeare's works and evaluate temperature-based sampling.
- **Corpus:** Public-domain TinyShakespeare corpus (Project Gutenberg). Text length: 1,115,394 characters; 65 unique characters.
- **Preprocessing:** Sliding window extraction (`seq_length=40`, `step=3`), bidirectional dictionary mappings (`char_to_idx`, `idx_to_char`).
- **Architecture:** `Input(40) -> Embedding(65, 64) -> LSTM(128) -> Dropout(0.2) -> Dense(65, softmax)`.
- **Temperature Sampling Formulation:**
  $$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
  - **$T = 0.4$ (Low):** Sharp, conservative, adheres to common Shakespeare dialogue tags (`FIRST CITIZEN:`) and repeated n-grams.
  - **$T = 0.8$ (Balanced):** Natural balance of vocabulary variety (*"thee", "thou", "hath"*) and syntactic validity.
  - **$T = 1.2$ (High):** Diverse and exploratory, with elevated entropy and occasional invented words.

---

### 03. Convolutional Denoising Autoencoder on Fashion-MNIST (`03_denoising_autoencoder_fashion_mnist.ipynb`)
- **Objective:** Train a self-supervised convolutional autoencoder to project noisy clothing images back onto the clean data manifold.
- **Dataset:** Fashion-MNIST (`keras.datasets.fashion_mnist`), 60,000 train, 10,000 test, $28 \times 28$ grayscale.
- **Corruption Process:** Zero-mean Gaussian noise ($\sigma = 0.35$), clipped to $[0.0, 1.0]$. The model is fed corrupted inputs targeting clean ground-truth images.
- **Architecture:**
  - **Encoder:** `Conv2D(32, (3,3), relu, same) -> MaxPool(2,2) -> Conv2D(16, (3,3), relu, same) -> MaxPool(2,2)` (Latent shape: $7 \times 7 \times 16 = 784$).
  - **Decoder:** `Conv2DTranspose(16, (3,3), strides=2, relu, same) -> Conv2DTranspose(32, (3,3), strides=2, relu, same) -> Conv2D(1, (3,3), sigmoid, same)` (Reconstructs $28 \times 28 \times 1$).
- **Loss:** Mean Squared Error (MSE).
- **Key Results:**
  - Input Noise MSE: $\approx 0.083$
  - Reconstructed Denoised MSE: $\approx 0.016$
  - Noise Error Reduction: $\approx 80.7\%$
  - Side-by-side visual comparison illustrates strong background noise elimination and silhouette recovery, with slight smoothing of fine fabric textures.

---

### 04. Generative Adversarial Network (GAN) on MNIST Digits (`04_gan_mnist_digit_generation.ipynb`)
- **Objective:** Implement a Deep Convolutional GAN (DCGAN) to synthesize handwritten digits from Gaussian latent noise.
- **Minimax Objective:**
  $$\min_G \max_D V(D, G) = \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$$
- **Dataset:** MNIST handwritten digits ($28 \times 28$), normalized to $[-1.0, 1.0]$.
- **Architectures:**
  - **Generator:** Latent vector $z \in \mathbb{R}^{100} \to$ `Dense(7*7*128) -> LeakyReLU(0.2) -> Reshape((7,7,128)) -> Conv2DTranspose(64, (4,4), s=2) -> LeakyReLU(0.2) -> Conv2DTranspose(1, (4,4), s=2, tanh)`.
  - **Discriminator:** Image $(28, 28, 1) \to$ `Conv2D(64, (3,3), s=2) -> LeakyReLU(0.2) -> Dropout(0.3) -> Conv2D(128, (3,3), s=2) -> LeakyReLU(0.2) -> Dropout(0.3) -> Flatten -> Dense(1, sigmoid)`.
- **Training Strategy:**
  - Dual Adam optimizers ($lr=0.0002, \beta_1=0.5$).
  - One-sided label smoothing ($0.9$ for real images) to prevent overconfident discriminator saturation.
  - Tracking fixed noise vectors across training epochs.
- **Key Observations:**
  - Epoch 1: Amorphous Gaussian noise and vague cloudy patches.
  - Epoch 4: Curvature emergence, loop primitives, and vertical strokes.
  - Epoch 12: Distinct digit morphologies (clear '0's, '1's, '7's, and '3's) on dark backgrounds.
  - Analysis explains why GAN losses oscillate rather than monotonically decrease.

---

### 05. Mini-Project: Interactive LSTM Text Generation (`05_lstm_text_generation_app.ipynb` & `app.py`)
- **Objective:** Complete end-to-end training, serialization, and interactive deployment of the Shakespeare character generator.
- **Shared Artifacts:**
  - Model file: `Week-3/models/text_generator.keras`
  - Vocabulary: `Week-3/models/vocabulary.json`
  - Configuration: `Week-3/models/model_config.json`
- **Application (`app.py`):**
  - Built with **Streamlit**.
  - Uses `@st.cache_resource` for instant, non-blocking page re-renders.
  - Controls: Custom seed text area, preset buttons, temperature slider ($0.1$ to $1.5$), generation length slider ($50$ to $500$ characters).
  - Robust error handling: Handles empty seed text, warns and safely strips out-of-vocabulary characters, and provides training instructions if model files are missing.

---

## 3. Verified Sample Outputs from Trained Model

**Seed Prompt:**
```
To be, or not to be, that is the question:
```

### Low Temperature ($T = 0.4$) — Strict & Conservative:
```
To be, or not to be, that is the question:
Whether the shall the shall the state of the state,
And the world the heart of the father and the son,
To see the court of the people and the prince.
```

### Balanced Temperature ($T = 0.8$) — Natural & Creative:
```
To be, or not to be, that is the question:
Which you have spoke of your brother's love,
And let me hear the voice of our revenge:
For thou art come to tell me of his death.
```

### High Temperature ($T = 1.2$) — High Entropy & Exploratory:
```
To be, or not to be, that is the question:
And yet, by'r lak'd, 'twill plud his quonch,
Nay, speak unchaff'd! Who's there, sirrah?
Thy wrondous crown is gash'd in Rome!
```

---

## 4. Setup, Installation & Verification

### Step 1: Install Dependencies
From the repository root or `Week-3/`:
```bash
pip install -r Week-3/requirements.txt
```

### Step 2: Run Automated Tests
```bash
pytest Week-3/tests/test_app.py -v
```
**Actual Verification Status:** `10 passed in 41.90s`.

### Step 3: Run the Interactive Streamlit Application
```bash
streamlit run Week-3/app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 5. Technical Limitations & Notes
1. **CPU Execution:** All models and training loops are designed and tuned for CPU execution with native Windows support. Training times are optimized to complete in minutes.
2. **Character-Level Horizon:** The character LSTM maintains coherent grammatical structure across 10–30 characters, but lacks higher-level plot coherence found in large transformer language models.
3. **GAN Mode Coverage:** The basic DCGAN demonstrates clear stroke and digit synthesis, but exhibits slight bias toward high-frequency digit classes (e.g., 0, 1, 7).
