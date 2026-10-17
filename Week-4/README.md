# Week 4: Advanced Deep Learning, Optimization, Deployment, and Major Capstone

Welcome to Week 4 of the Deep Learning curriculum. This module covers advanced deep learning concepts, optimization strategies, lightweight edge deployment, production REST API serving, and a major capstone project: **Document Understanding & Layout Classification**.

---

## 1. Directory Structure

```text
Week-4/
├── 01_self_attention_from_scratch.ipynb    # Task A: Pure PyTorch Self-Attention from Scratch
├── 02_model_optimization.ipynb             # Task B: BatchNorm, Dropout & LR Scheduling on CIFAR-10
├── 03_tensorflow_lite_benchmark.ipynb      # Task C: Keras to TFLite & INT8 Quantization Benchmark
├── 04_model_rest_api.ipynb                 # Task D: FastAPI Model Deployment & Automated Tests
├── requirements.txt                        # Week 4 dependencies
├── README.md                               # This comprehensive documentation
└── major_project/                          # Capstone: Document Understanding & Layout Classification
    ├── backend/
    │   ├── app.py                          # FastAPI lifespan, CORS, and application entrypoint
    │   ├── routes.py                       # Route handlers (/health, /classes, /predict)
    │   ├── schemas.py                      # Pydantic request & response validation schemas
    │   └── model_utils.py                  # Model caching, image preprocessing, and inference
    ├── frontend/
    │   ├── app.py                          # Streamlit UI with 1-click presets & dual inference
    │   └── requirements.txt                # Frontend dependencies
    ├── training/
    │   ├── generate_dataset.py             # Deterministic synthetic document layout generator
    │   ├── preprocessing.py                # Preprocessing, normalization, and dataset splitting
    │   ├── train.py                        # MobileNetV2 + Spatial Head transfer learning pipeline
    │   └── evaluate.py                     # Held-out evaluation, confusion matrix & ROC curves
    ├── models/
    │   ├── document_classifier.keras       # Trained Keras model artifact
    │   ├── class_indices.json              # Label mapping dictionary
    │   ├── model_metadata.json             # Hyperparameters, metrics, and training metadata
    │   └── test_data.npy                   # Saved held-out test split for reproducibility
    ├── reports/
    │   ├── confusion_matrix.png            # Generated test confusion matrix plot
    │   ├── roc_curve.png                   # Multi-class One-vs-Rest ROC curve plot
    │   └── evaluation_metrics.json         # Exact test metrics (loss, accuracy, F1, latency)
    ├── tests/
    │   ├── test_preprocessing.py           # Unit tests for image resizing, splits & normalization
    │   ├── test_model.py                   # Tests for model weights, shapes, and test accuracy
    │   └── test_api.py                     # Integration tests for FastAPI endpoints via TestClient
    ├── docs/
    │   ├── architecture.md                 # System architecture and end-to-end dataflow
    │   ├── methodology.md                  # Layout synthesis, spatial preservation & training
    │   └── results.md                      # Quantitative benchmark, metrics, and error analysis
    ├── Dockerfile                          # Production multi-service Docker container
    ├── docker-compose.yml                  # Docker Compose orchestration (API + Frontend)
    ├── .dockerignore                       # Exclusions for container builds
    ├── .gitignore                          # Git exclusions
    ├── requirements.txt                    # Major project dependencies
    └── README.md                           # Capstone project documentation
```

---

## 2. Learning Exercises Overview

### Task A — Self-Attention From Scratch in PyTorch
- **File:** `01_self_attention_from_scratch.ipynb`
- **Objective:** Understand and implement scaled dot-product self-attention explicitly without using built-in high-level layers (`torch.nn.MultiheadAttention`).
- **Mathematical Formulation:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Key Components:**
  - Learnable linear projections: $W_Q, W_K, W_V, W_O$.
  - Explicit scaling by $\frac{1}{\sqrt{d_k}}$ to prevent saturated softmax gradients.
  - Softmax along the key dimension ($\text{dim} = -1$).
  - Rigorous assertions confirming output dimensions $(B, T, d_{\text{model}})$, weight dimensions $(B, T, T)$, and probability normalization ($\sum_j A_{ij} = 1.0 \pm 10^{-5}$).
  - Interactive Seaborn heatmap demonstrating token-to-token contextualization.
  - Equivalence check against PyTorch's native `F.scaled_dot_product_attention` ($\Delta < 10^{-6}$).

---

### Task B — Deep Learning Model Optimization
- **File:** `02_model_optimization.ipynb`
- **Execution Status:** **Fully Executed** (Generated `optimization_curves.png`)
- **Objective:** Systematically isolate and measure the effects of Batch Normalization, Dropout, and Learning Rate Scheduling on CNN convergence and generalization using CIFAR-10.
- **Configurations Compared:**
  1. **Baseline CNN:** Standard Conv-ReLU-Pool blocks with fixed learning rate ($\text{lr} = 10^{-3}$).
  2. **CNN + Batch Normalization:** Stabilizes internal covariate shifts and accelerates gradient propagation.
  3. **CNN + Dropout ($p = 0.3$):** Breaks redundant feature co-adaptation and narrows the generalization gap.
  4. **CNN + LR Schedule:** `ReduceLROnPlateau` decays learning rate by $0.5$ when validation loss stagnates.
  5. **Combined Architecture:** Synergistic combination of BatchNorm + Dropout + LR Scheduling.
- **Key Takeaways & Measured Results:**
  - Batch normalization accelerates early convergence dramatically compared to baseline.
  - Dropout reduces training overconfidence, narrowing the generalization gap between train and validation.
  - The combined model delivers the most robust convergence curve and highest test accuracy.

---

### Task C — TensorFlow Lite Conversion & Benchmarking
- **File:** `03_tensorflow_lite_benchmark.ipynb`
- **Execution Status:** **Fully Executed** (Measured on-disk footprints and Keras latency)
- **Objective:** Convert a trained Keras classification model to TensorFlow Lite and benchmark inference latency, model footprint, and accuracy retention.
- **Quantization Modes:**
  1. **Original Keras (`.keras`):** Baseline floating-point graph.
  2. **TFLite FP32:** FlatBuffer serialized model preserving full 32-bit floating point precision.
  3. **TFLite Dynamic Range INT8 Quantization (`tf.lite.Optimize.DEFAULT`):** 8-bit integer weights with on-the-fly floating point activation dequantization.
- **Actual Measured Results:**
  | Metric | Keras (Original) | TFLite FP32 | TFLite INT8 Quant |
  | :--- | :---: | :---: | :---: |
  | **File Size** | 1,277.3 KB | **417.9 KB (3.1x)** | **111.6 KB (11.4x)** |
  | **Test Accuracy** | 87.50% | N/A* | N/A* |
  | **Median Latency** | 6.20 ms | N/A* | N/A* |
  | **Throughput** | 159.7 FPS | N/A* | N/A* |

  *\* Runtime Environment Limitation Note:*
  Model conversion and quantization executed successfully on disk, demonstrating up to 11.4x compression. However, direct in-process execution via `tf.lite.Interpreter` on this host environment encountered an upstream C-API wrapper ABI error (`AttributeError: _ARRAY_API not found` / `SystemError: CreateWrapperFromFile`). The installed Windows binary for TensorFlow 2.21 was built against NumPy 1.x C-API, whereas the environment runs NumPy 2.4.6. As recommended by official TensorFlow guidance, running TFLite inference in production should either utilize NumPy < 2 or the modern `ai_edge_litert` runtime.

---

### Task D — Model REST API Serving with FastAPI
- **File:** `04_model_rest_api.ipynb`
- **Execution Status:** **Fully Executed** (TestClient integration tests verified)
- **Objective:** Implement a production-grade REST API serving deep learning models with FastAPI.
- **Architecture Highlights:**
  - Model weights loaded once at startup via `@asynccontextmanager` (lifespan), avoiding per-request I/O latency.
  - Health check endpoint (`GET /health`) with status, model availability, and version.
  - Inference endpoint (`POST /predict`) with multipart image upload and configurable `top_k`.
  - Input validation: rejects unsupported extensions (HTTP 400), empty payloads (HTTP 400), and files exceeding size limits (HTTP 413).
  - Automated test suite using `starlette.testclient.TestClient`.

---

## 3. Major Capstone: Document Understanding & Layout Classification

### 3.1 Problem Definition & Justification
In automated enterprise workflows, inbound documents must be routed before OCR or field extraction can occur. We frame this as a 5-class visual layout classification task:
1. **`invoice`:** Vendor header, billing columns, itemized price tables, total balance.
2. **`receipt`:** Narrow vertical slip, store metadata, column items, barcode.
3. **`form`:** Dense field grids, checkbox matrices, personal data lines, signature areas.
4. **`letter`:** Date header, recipient address, salutation, body prose paragraphs, signature.
5. **`report`:** Two-column academic/corporate layout, section headers, figure/chart placeholders.

Visual layout classification is language-agnostic, low-latency (<30 ms), and operates effectively on low-resolution scans without requiring expensive OCR engines.

---

### 3.2 Preprocessing & Backbone Normalization Alignment
- **Expected Backbone Representation:** ImageNet-pretrained MobileNetV2 expects normalized inputs in $[-1.0, 1.0]$.
- **In-Graph Preprocessing:** Rather than relying on fragile distributed client-side normalization, the Keras model embeds `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` as layer 1. Raw float32 pixel values in $[0.0, 255.0]$ passed from `training/preprocessing.py`, `backend/model_utils.py`, or test scripts are uniformly rescaled within the computational graph.
- **Spatial Preservation:** Instead of `GlobalAveragePooling2D()` (which collapses spatial coordinates and degrades document layout discrimination), we use:
  - $1 \times 1$ Conv2D (128 filters, ReLU, same padding)
  - `BatchNormalization`
  - `Flatten` (preserving the $7 \times 7 \times 128 = 6,272$ feature units)
  - `Dense` (64, ReLU)
  - `Dropout` (0.3)
  - `Dense` (5, Softmax)

---

### 3.3 Quantitative Evaluation on Held-Out Test Set
Evaluated strictly on an isolated held-out test split ($N = 26$, ~17% of total dataset, 5 forms, 5 invoices, 5 letters, 6 receipts, 5 reports):

| Metric | Measured Value |
| :--- | :---: |
| **Test Accuracy** | **100.0% (26 / 26 correct)** |
| **Categorical Cross-Entropy Loss** | **0.0000** |
| **Macro Precision** | **1.0000** |
| **Macro Recall** | **1.0000** |
| **Macro F1-Score** | **1.0000** |
| **One-vs-Rest ROC AUC** | **1.0000** |
| **Average Inference Latency** | **~28 ms (CPU)** |

> [!WARNING]
> **Methodological Reality Check:** Achieving 100% accuracy on 26 synthetic test images does **NOT** establish reliable performance on real-world document understanding. The synthetic generator creates clean, structured documents with prominent geometric invariants (barcodes, tables, signatures). Real-world document classification (e.g., RVL-CDIP) must deal with heavy scanning noise, physical crumpling, faded thermal ink, multi-page variations, and hybrid layouts, where accuracy would be significantly lower without domain adaptation.

Generated artifacts in `major_project/reports/`:
- `confusion_matrix.png`: Clean diagonal confusion matrix across all 26 test samples.
- `roc_curve.png`: Multi-class One-vs-Rest ROC curves.
- `evaluation_metrics.json`: Exported numerical metrics.

---

## 4. Running the Application

### 4.1 Local Execution

1. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start the FastAPI REST Service:**
   ```powershell
   cd major_project
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Healthcheck: `http://localhost:8000/health`
   - Interactive Swagger Docs: `http://localhost:8000/docs`

3. **Start the Streamlit User Interface:**
   ```powershell
   streamlit run frontend/app.py --server.port 8501
   ```
   Open `http://localhost:8501` to test with 1-click sample presets or upload custom document scans.

---

### 4.2 Dockerized Deployment

The capstone includes a multi-service Docker configuration (`Dockerfile` and `docker-compose.yml`):

```bash
# Build and launch both API and Frontend containers
docker compose up --build -d

# Check service logs
docker compose logs -f

# Verify API health
curl http://localhost:8000/health

# Stop containers
docker compose down
```

> [!NOTE]
> During local validation, Docker CLI (`docker.exe`) was present on disk while the local Docker Desktop daemon service was inactive. All Dockerfiles, compose files, and container configs have been written and statically validated.

---

## 5. Automated Test Suite

Run the automated test suite with pytest:

```powershell
pytest major_project/tests -v
```

**Verification Results:**
- `test_preprocessing.py`: 3/3 passed (dataset loading, non-overlapping splits, image normalization).
- `test_model.py`: 5/5 passed (checkpoint existence, input/output tensor shapes, probability sum to 1.0, class consistency, test accuracy $\ge 0.80$).
- `test_api.py`: 6/6 passed (`/health`, `/classes`, `/predict` valid upload, invalid file extension, corrupted stream, `top_k` ranking).
- **Total: 14 passed in 34.12s.**
