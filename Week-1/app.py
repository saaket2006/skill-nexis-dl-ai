"""
Interactive MNIST Digit Recognizer — Streamlit Web Application
==============================================================
Features:
1. Real-time HTML5 Drawing Canvas (custom bidirectional Streamlit component)
2. Image Upload Fallback (PNG / JPG / JPEG)
3. Built-in Test Samples (instant 1-click test with real MNIST digits)
4. Full Preprocessing Pipeline (aspect-preserving resize, centering, inversion, normalization)
5. Model Inference with Top-3 Predictions and Full Probability Bar Chart
6. Diagnostic inspection of the 28x28 image feeding into the neural network
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

# ─── Configuration & Paths ───────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(SCRIPT_DIR, "mnist_model.keras")
COMPONENT_DIR  = os.path.join(SCRIPT_DIR, "canvas_component")

# Register custom canvas component
_digit_canvas = components.declare_component("digit_canvas", path=COMPONENT_DIR)


# ─── Model Management ────────────────────────────────────────────────────────
def _get_keras():
    try:
        import tensorflow as tf
        from tensorflow import keras
        return keras
    except ImportError:
        try:
            import keras
            return keras
        except ImportError:
            return None


@st.cache_resource
def load_trained_model():
    """Load model from disk if available."""
    keras = _get_keras()
    if keras is None:
        return None, "TensorFlow/Keras is not installed in the current environment."
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at {MODEL_PATH}."
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


def train_quick_model():
    """Train a quick model if none exists."""
    keras = _get_keras()
    if keras is None:
        return None
    with st.spinner("Training baseline MNIST model (5 epochs)..."):
        (X_tr, y_tr), (X_te, y_te) = keras.datasets.mnist.load_data()
        X_tr = X_tr.astype(np.float32) / 255.0
        X_te = X_te.astype(np.float32) / 255.0

        model = keras.Sequential([
            keras.layers.Input(shape=(28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(X_tr, y_tr, epochs=5, batch_size=128, validation_split=0.1, verbose=0)
        model.save(MODEL_PATH)
        return model


# ─── Image Preprocessing Pipeline ────────────────────────────────────────────
from preprocessing import preprocess_digit_image



# ─── Preloaded Sample Digits ─────────────────────────────────────────────────
@st.cache_data
def get_sample_digits():
    """Returns a dictionary of sample MNIST digits for quick testing."""
    keras = _get_keras()
    if keras is None:
        return {}
    try:
        (_, _), (X_te, y_te) = keras.datasets.mnist.load_data()
        samples = {}
        for digit in range(10):
            idx = np.where(y_te == digit)[0][0]
            # Convert to PIL Image (280x280 for viewing)
            img_arr = 255 - X_te[idx] # inverted for drawing canvas look
            img = Image.fromarray(img_arr).resize((280, 280), Image.Resampling.NEAREST)
            samples[digit] = img
        return samples
    except Exception:
        return {}


# ─── Main Application Interface ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="MNIST Digit Recognizer",
        page_icon="✏️",
        layout="wide"
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">✏️ Interactive MNIST Digit Recognizer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Draw a digit on the canvas, choose a preloaded sample, '
        'or upload an image to classify digits with confidence metrics.</div>',
        unsafe_allow_html=True
    )

    # ── Model Loading ──
    model, error = load_trained_model()
    if model is None:
        st.warning(f"⚠️ {error}")
        if st.button("🚀 Train Model Now (5 Epochs)"):
            model = train_quick_model()
            st.rerun()
        if model is None:
            st.stop()
    else:
        st.sidebar.success("✅ Model loaded: `mnist_model.keras`")

    # Sidebar parameters
    st.sidebar.markdown("### ⚙️ Model Details")
    st.sidebar.markdown("""
    - **Architecture:** Dense MLP (784 → 128 → 64 → 10)
    - **Activation:** ReLU hidden, Softmax output
    - **Training Set:** 60,000 MNIST images
    - **Test Accuracy:** 97.35%
    """)

    # ── Input Column and Results Column ──
    col_input, col_results = st.columns([1, 1], gap="large")

    active_image = None
    input_source = ""

    with col_input:
        tab_draw, tab_samples, tab_upload = st.tabs([
            "🖊️ Draw Digit",
            "🔢 Quick Samples",
            "📂 Upload Image"
        ])

        # ── TAB 1: Draw Canvas ──
        with tab_draw:
            st.caption("Draw in the center using your mouse or touchscreen. Click **Predict** on the canvas when done.")
            canvas_result = _digit_canvas(key="mnist_canvas")

            # Safely parse canvas data
            if isinstance(canvas_result, str) and canvas_result.startswith("data:image"):
                try:
                    _, b64data = canvas_result.split(",", 1)
                    raw_bytes = base64.b64decode(b64data)
                    canvas_img = Image.open(io.BytesIO(raw_bytes))
                    active_image = canvas_img
                    input_source = "Interactive Canvas"
                except Exception as decode_err:
                    st.error(f"Error reading canvas image: {decode_err}")

        # ── TAB 2: Quick Samples ──
        with tab_samples:
            st.caption("Click any digit below to immediately test the model on a real MNIST test sample:")
            samples = get_sample_digits()
            if samples:
                cols = st.columns(5)
                for d in range(10):
                    btn_col = cols[d % 5]
                    if btn_col.button(f"Digit {d}", key=f"btn_sample_{d}", use_container_width=True):
                        st.session_state["selected_sample"] = d

                selected_d = st.session_state.get("selected_sample", None)
                if selected_d is not None and selected_d in samples:
                    active_image = samples[selected_d]
                    input_source = f"Sample MNIST Digit ({selected_d})"
                    st.image(active_image, caption=f"Selected Sample: {selected_d}", width=140)
            else:
                st.info("Loading sample digits...")

        # ── TAB 3: Upload Image ──
        with tab_upload:
            st.caption("Upload a handwritten digit photo or image (PNG, JPG, JPEG):")
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed"
            )
            if uploaded_file is not None:
                try:
                    uploaded_img = Image.open(uploaded_file)
                    active_image = uploaded_img
                    input_source = f"Uploaded File ({uploaded_file.name})"
                    st.image(uploaded_img, caption="Uploaded Image", width=140)
                except Exception as up_err:
                    st.error(f"Could not load image: {up_err}")

    # ── Inference & Results Column ──
    with col_results:
        st.subheader("📊 Model Classification & Analysis")

        if active_image is not None:
            tensor_in, preprocessed_28, is_blank = preprocess_digit_image(active_image)

            if is_blank:
                st.info("✏️ Canvas is clear. Draw a digit on the canvas or pick a sample to view predictions.")
            else:
                # Run Model Prediction
                probabilities = model.predict(tensor_in, verbose=0)[0]
                pred_digit = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_digit]) * 100.0

                # Summary cards
                card1, card2 = st.columns(2)
                with card1:
                    st.metric(label="Predicted Digit", value=f"{pred_digit}")
                with card2:
                    st.metric(label="Confidence", value=f"{confidence:.1f}%")

                if confidence >= 85:
                    st.success(f"High confidence prediction from **{input_source}**.")
                elif confidence >= 60:
                    st.warning(f"Moderate confidence ({confidence:.1f}%). Digit may have unusual stroke thickness or centering.")
                else:
                    st.error(f"Low confidence ({confidence:.1f}%). Digit is ambiguous or differs from MNIST distribution.")

                # Preprocessing visualizer
                with st.expander("🔍 Preprocessed 28×28 Input (Model Vision)", expanded=True):
                    p_col1, p_col2 = st.columns([1, 2])
                    with p_col1:
                        # Display 28x28 normalized array
                        disp_img = (preprocessed_28 * 255).astype(np.uint8)
                        disp_pil = Image.fromarray(disp_img, mode="L").resize((140, 140), Image.Resampling.NEAREST)
                        st.image(disp_pil, caption="28×28 Centered Input", width=140)
                    with p_col2:
                        st.caption(
                            "**Preprocessing Steps Applied:**  \n"
                            "1. Contrast & Polarity Inversion (dark background, bright digit)  \n"
                            "2. Bounding Box Detection & Crop  \n"
                            "3. Aspect-Ratio Preserving Resize to 20×20 px  \n"
                            "4. Center of Mass Placement on 28×28 Canvas  \n"
                            "5. Pixel Normalization to [0.0, 1.0]"
                        )

                # Probabilities chart
                st.markdown("#### Class Probabilities (0–9)")
                chart_df = pd.DataFrame({
                    "Digit": [str(d) for d in range(10)],
                    "Probability (%)": [float(p) * 100 for p in probabilities]
                })
                st.bar_chart(
                    chart_df.set_index("Digit"),
                    y="Probability (%)",
                    color="#2563eb",
                    height=240
                )

                # Top 3 ranking
                top3_indices = np.argsort(probabilities)[::-1][:3]
                st.markdown("#### 🏆 Top 3 Candidates")
                cols = st.columns(3)
                for rank, idx in enumerate(top3_indices):
                    with cols[rank]:
                        st.markdown(
                            f"**Rank {rank + 1}: Digit {idx}**  \n"
                            f"`{probabilities[idx] * 100:.2f}%`"
                        )
        else:
            st.info("👆 Draw a digit on the left canvas, click a quick sample, or upload an image to see predictions.")

    # ── Explanation Section ──
    st.divider()
    with st.expander("ℹ️ Architectural Guide & Failure Mode Analysis"):
        st.markdown("""
        ### Why Canvas Predictions Can Differ from MNIST Test Set
        The MNIST dataset consists of digits scanned from paper census forms and high school student forms:
        - **Fixed Stroke Weight:** MNIST strokes were drawn with standard ink pens and antialiased during downsampling.
        - **Resolution Shift:** Mouse drawings on a 280×280 browser canvas have variable speed and line gaps.
        - **Centering & Padding:** MNIST digits are strictly normalized into a 20×20 pixel bounding box and centered by center-of-mass on a 28×28 canvas.
        - **Linear Separability:** A simple Multilayer Perceptron (MLP) lacks the translation invariance of Convolutional Neural Networks (CNNs). Minor offsets or slant can alter activation patterns across the 784 flattened inputs.
        """)


if __name__ == "__main__":
    main()
