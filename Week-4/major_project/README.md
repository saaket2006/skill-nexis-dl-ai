# Week 4 Major Project: Document Understanding & Layout Classification

An end-to-end Deep Learning system for classifying business document images by their layout structure into 5 distinct categories: **Invoices**, **Receipts**, **Forms**, **Letters**, and **Reports**.

The project integrates transfer learning (MobileNetV2 feature backbone with a spatial classification head), a FastAPI REST service, an interactive Streamlit UI, automated testing, and containerized deployment with Docker and Docker Compose.

---

## 1. Project Overview

Business document automation starts with understanding what type of document has been captured before OCR or information extraction begins. Document images vary drastically in structure:
- **Invoice**: Prominent vendor header, bill-to block, itemized line tables, invoice numbers, and total balance.
- **Receipt**: Narrow vertical slip, top store info, columnar prices, transaction timestamp, and barcode.
- **Form**: Grid-like structured fields, checkbox matrices, applicant details, and signature boxes.
- **Letter**: Formal sender/recipient letterhead, date, structured multi-paragraph narrative, and sign-off.
- **Report**: Multi-column analytic summaries, chart/figure placeholders, executive summaries, and section headers.

---

## 2. Directory Structure

```text
major_project/
├── backend/
│   ├── app.py                # FastAPI lifecycle, CORS, and server entrypoint
│   ├── routes.py             # Route handlers (/health, /classes, /predict)
│   ├── schemas.py            # Pydantic request & response models
│   └── model_utils.py        # Model loading, image preprocessing, and inference
├── frontend/
│   ├── app.py                # Streamlit user interface with live API and local fallback
│   └── requirements.txt      # Frontend-specific dependencies
├── training/
│   ├── generate_dataset.py   # Deterministic layout generator creating 150 document samples
│   ├── preprocessing.py      # Resizing, normalization, data splitting (train/val/test)
│   ├── train.py              # Transfer learning training pipeline with MobileNetV2
│   └── evaluate.py           # Evaluation on held-out test set (Confusion matrix, ROC curves)
├── models/
│   ├── document_classifier.keras  # Saved trained Keras model
│   ├── class_indices.json         # Label mapping dictionary
│   ├── model_metadata.json        # Training parameters & metadata
│   └── test_data.npy              # Held-out test split for reproducibility
├── reports/
│   ├── confusion_matrix.png       # Test set confusion matrix
│   ├── roc_curve.png              # Multi-class One-vs-Rest ROC curves
│   └── evaluation_metrics.json    # Test loss, accuracy, precision, recall, F1
├── tests/
│   ├── test_preprocessing.py      # Preprocessing & split integrity tests
│   ├── test_model.py              # Model artifact and inference tests
│   └── test_api.py                # REST API endpoint tests with TestClient
├── docs/
│   ├── architecture.md            # System architecture and data flow
│   ├── methodology.md             # Dataset synthesis, augmentation & modeling rationale
│   └── results.md                 # Quantitative evaluation and benchmarking results
├── Dockerfile                     # Multi-service production Docker container
├── docker-compose.yml             # Orchestration for FastAPI API and Streamlit UI
├── requirements.txt               # Complete Python dependencies
└── README.md                      # This documentation
```

---

## 3. Dataset & Preprocessing

- **Total Samples:** 150 high-resolution document images (30 per class).
- **Split Distribution:**
  - **Train:** 105 images (70%, 21 per class)
  - **Validation:** 19 images (~13%)
  - **Held-Out Test:** 26 images (~17%)
- **Preprocessing Pipeline & Normalization Alignment:**
  1. Input image decoded and converted to 3-channel RGB.
  2. Bilinear resize to $224 \times 224$ pixels.
  3. Standardized to float32 $[0.0, 255.0]$.
  4. In-Graph Rescaling: The Keras model embeds `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` as layer 1, mapping values to $[-1.0, 1.0]$ matching ImageNet MobileNetV2 pretraining conventions across all training, evaluation, and REST API calls.

> [!WARNING]
> **Methodological Note on Synthetic Evaluation:** The model achieves 100% accuracy on the 26 held-out synthetic test images because `training/generate_dataset.py` generates clean, highly structured geometric templates. This performance demonstrates that the network has perfectly mastered the procedural layout motifs (e.g. barcodes, tables, headers), but does **NOT** prove generalization to real-world scanned, noisy, or crumpled documents.

---

## 4. Model Architecture & Training Results

- **Backbone:** MobileNetV2 pretrained on ImageNet (weights frozen).
- **Embedded Preprocessing:** `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` at index 0.
- **Spatial Head:** $1 \times 1$ Conv2D (128 filters) $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ Flatten ($7 \times 7 \times 128 = 6272$ units) $\rightarrow$ Dense (64, ReLU) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (5, Softmax).
- **Optimizer:** Adam ($\text{lr} = 10^{-3}$, reduced on plateau).
- **Loss:** Categorical Cross-Entropy.
- **Measured Test Metrics (Held-Out Test Set, $N=26$):**
  - **Test Accuracy:** 100.0% (26 / 26 correct)
  - **Test Loss:** 0.0000
  - **Macro Precision:** 1.0000
  - **Macro Recall:** 1.0000
  - **Macro F1-Score:** 1.0000
  - **Multi-class One-vs-Rest AUC:** 1.0000
  - **Average Latency:** ~28 ms per document (Intel CPU)

---

## 5. Quickstart & Usage

### A. Run Locally

1. **Activate Python Environment:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Start the FastAPI REST Service:**
   ```powershell
   uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
   ```
   - Healthcheck: `http://localhost:8000/health`
   - Interactive Swagger API docs: `http://localhost:8000/docs`

3. **Start the Streamlit Web Application:**
   ```powershell
   streamlit run frontend/app.py --server.port 8501
   ```
   Open `http://localhost:8501` in your browser.
   - The UI provides an explicit **Execution Mode** selector:
     - `REST API (Microservice)`: Sends requests to `http://localhost:8000/predict`. Displays an explicit error banner if the API is offline; no silent fabricated fallback.
     - `Standalone Direct Model`: Evaluates the model directly in-process via `backend/model_utils.py`.
   - Every inference shows its active execution source badge.

---

### B. Run with Docker & Docker Compose

```bash
# Build and start both API and Frontend containers
docker compose up --build -d

# Check service logs
docker compose logs -f

# Verify API health
curl http://localhost:8000/health

# Stop containers
docker compose down
```

> [!NOTE]
> **Docker Validation Status:** Docker CLI version 29.6.2 is installed. The Docker Desktop Linux engine daemon was offline on the host system (`//./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`). Container files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) have been verified for syntax and path correctness; live container execution remains unverified due to the daemon state.

---

## 6. REST API Endpoints

### `GET /health`
Returns service status, model loading state, and active classes.

### `GET /classes`
Returns the list of 5 recognized document layout categories.

### `POST /predict`
Submits a document image file for classification.
- **Form Data:**
  - `file`: Document image file (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp`)
  - `top_k`: Integer (default: 3, max: 5)
- **Sample JSON Response:**
  ```json
  {
    "predicted_category": "invoice",
    "confidence": 0.9842,
    "confidence_percentage": "98.42%",
    "top_k": [
      {"category": "invoice", "confidence": 0.9842, "percentage": "98.42%"},
      {"category": "receipt", "confidence": 0.0115, "percentage": "1.15%"},
      {"category": "report", "confidence": 0.0028, "percentage": "0.28%"}
    ],
    "inference_time_ms": 26.4,
    "filename": "invoice_sample.png"
  }
  ```

---

## 7. Automated Testing

Run the complete test suite with `pytest`:

```powershell
pytest tests/ -v
```

Tests cover:
1. `test_preprocessing.py`: Image resizing, split non-overlap, and normalization.
2. `test_model.py`: Model checkpoint loading, input/output tensor shapes, probability normalization, and test accuracy verification.
3. `test_api.py`: FastAPI endpoints (`/health`, `/classes`, `/predict`), file validation, error codes, and top-k ranking.
