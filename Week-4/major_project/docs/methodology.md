# Methodology: Document Understanding & Layout Classification

This document outlines the design decisions, dataset generation, model selection, loss formulation, and evaluation protocol for the Week 4 Document Understanding Capstone.

---

## 1. Problem Formulation & Category Definition

Document layout classification frames the problem of identifying document type purely from visual/structural layout cues rather than text OCR. This approach has key advantages:
1. **Language-agnostic:** Operates equally well on scanned forms or multi-lingual documents without requiring heavy language models or OCR engines.
2. **Low Latency:** Inference runs in tens of milliseconds rather than seconds.
3. **Resilience to Poor OCR:** Works even when scanned text is low-resolution, blurred, or noisy.

We define 5 distinct structural classes:
- **`invoice`**: Characterized by a vendor header, two-column billing address block, bordered multi-row pricing table, and right-aligned grand total.
- **`receipt`**: Long, narrow vertical strip, centered store title, dotted separator lines, narrow column items, and bottom barcode.
- **`form`**: Dense grid of text entry fields, square checkboxes, bracketed personal data fields, and signature lines.
- **`letter`**: Top right/left date and address, salutation, left-aligned prose blocks with margins, and handwritten cursive signature.
- **`report`**: Two-column academic/corporate layout, section headers, figure/chart placeholders, and footer page numbering.

---

## 2. Dataset Synthesis, Generation Details, & Leakage Analysis

To ensure complete reproducibility without external copyright or network dependencies, `training/generate_dataset.py` synthesizes documents using PIL drawing primitives:
- **Total Dataset:** 150 high-resolution images (30 per class).
- **Class Characteristics & Visual Distinctions:**
  - **`invoice`**: Header banner, vendor block, two-column billing/shipping address, bordered multi-row tabular pricing grid, and right-aligned grand total.
  - **`receipt`**: Rendered on a narrow vertical aspect ratio ($360 \times 720$), centered store title, dotted divider lines, itemized single-column costs, and bottom vertical barcode stripes.
  - **`form`**: Multi-section enclosed boxes, bracketed entry fields (`[        ]`), square checkboxes, personal data labels (`Full Name:`, `Date of Birth:`), and signature boxes.
  - **`letter`**: Top sender/recipient address block, formal salutation (`Dear Sir/Madam,`), left-aligned multi-paragraph prose blocks with standard margins, valediction (`Sincerely,`), and simulated cursive vector signature.
  - **`report`**: Two-column layout with center gutter, bold section headers (`1.0 Executive Summary`), simulated vector bar/pie chart placeholder, and footer pagination.
- **Split Distribution & Partitioning:**
  - **Training Set (70%):** 105 images (21 per class)
  - **Validation Set (~13%):** 19 images
  - **Held-Out Test Set (~17%):** 26 images
  - Splits are generated deterministically using stratified sampling with `seed=42`. All feature extraction, scaling, and hyperparameter choices are locked prior to held-out test evaluation.

### 2.1 Template Sharing, Visual Homogeneity, & Leakage Assessment
A thorough methodological audit of the synthetic generator reveals:
1. **Geometric Invariants:** While line coordinates, row counts, and text strings vary randomly across seeds, each class relies on invariant visual motifs (e.g., all receipts contain bottom barcodes; all invoices contain grid tables; all letters have paragraph blocks).
2. **Artificial Separability:** The inter-class variance is dramatic and unobstructed by physical scanning artifacts (folds, wrinkles, perspective skew, sensor noise, variable illumination).
3. **Generalization Limitation:** 
   > [!WARNING]
   > 100% accuracy on 26 synthetic test images demonstrates that the model successfully learned the procedural rules of `generate_dataset.py`. It **does NOT** demonstrate real-world document understanding. Performance on real-world datasets (e.g., RVL-CDIP, Tobacco3482, or DocBank) would encounter noisy scans, multilingual scripts, and ambiguous multi-page structures, resulting in lower baseline accuracy without domain-specific fine-tuning.

---

## 3. Preprocessing & Model Architecture

### 3.1 Preprocessing Pipeline & Backbone Normalization
MobileNetV2 was pretrained on ImageNet under the standard convention:
$$\text{Pixel}_{\text{norm}} = \frac{\text{Pixel}}{127.5} - 1.0 \in [-1.0, 1.0]$$
To prevent silent inference bugs or pipeline discrepancies:
- **Input Standardization (`training/preprocessing.py`, `backend/model_utils.py`):** Images are decoded from RGB, resized with bilinear interpolation to $224 \times 224$, and converted to float32 in $[0.0, 255.0]$.
- **In-Graph Rescaling Layer:** The Keras model embeds `layers.Rescaling(scale=1.0/127.5, offset=-1.0)` as layer 1. Every forward pass—whether during training, offline evaluation, or live REST inference—executes identical normalization inside the computation graph.

### 3.2 Model Architecture: Transfer Learning with Spatial Preservation
We selected **MobileNetV2** (lightweight, mobile-ready, low parameter count) followed by a spatial compression head:
1. **Backbone:** MobileNetV2 pretrained on ImageNet (weights frozen to preserve low-level edge, corner, and line features).
2. **Spatial Feature Preservation:** Rather than applying `GlobalAveragePooling2D()` (which collapses document coordinates and destroys critical spatial hierarchy such as headers at the top vs. barcodes at the bottom), we employ a $1 \times 1$ Conv2D layer (128 filters, ReLU, same padding) to reduce channel depth while preserving the $7 \times 7$ grid.
3. **Batch Normalization:** Stabilizes feature activations across training batches.
4. **Flattening:** Unrolls the $7 \times 7 \times 128$ feature map to 6,272 units, retaining coordinate sensitivity (top-left vs. bottom-right features).
5. **Dense Bottleneck:** 64 units with ReLU activation.
6. **Regularization:** Dropout layer ($p = 0.3$) to prevent co-adaptation.
7. **Classification Head:** 5-unit Dense layer with Softmax activation.

---

## 4. Hyperparameter Tuning & Training Strategy

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Input Shape** | $224 \times 224 \times 3$ | Standard ImageNet resolution; preserves layout geometry |
| **Normalization** | Embedded Rescaling $[-1.0, 1.0]$ | Matches MobileNetV2 ImageNet pretraining |
| **Optimizer** | Adam ($\text{lr} = 10^{-3}$) | Rapid initial convergence with adaptive moment estimation |
| **Learning Rate Schedule** | `ReduceLROnPlateau` | Factor 0.5, patience 3, minimum $\text{lr} = 10^{-6}$ |
| **Batch Size** | 16 | Suitable for compact dataset size and memory footprint |
| **Epochs** | 12 | Complete convergence on synthetic dataset |
| **Loss Function** | Categorical Cross-Entropy | Multi-class mutual exclusivity |
| **Regularization** | Dropout 0.3 + BatchNorm | Mitigates overfitting on small layout datasets |
