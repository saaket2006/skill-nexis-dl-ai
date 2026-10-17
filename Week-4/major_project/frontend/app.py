"""
frontend/app.py
===============
Interactive Streamlit interface for Document Understanding & Layout Classification.
Connects directly to the FastAPI REST backend service by default.
Features explicit execution mode selection (REST API vs. Standalone Direct Model),
sample document presets, custom uploads, layout explainability, and transparent error handling.
"""

import os
import io
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import requests
import streamlit as st
from PIL import Image

# Setup page layout and branding
st.set_page_config(
    page_title="Document Understanding & Classification",
    page_icon="📄",
    layout="wide",
)

FRONTEND_DIR = Path(__file__).resolve().parent
MAJOR_PROJECT_DIR = FRONTEND_DIR.parent
DATA_DIR = MAJOR_PROJECT_DIR / "data" / "documents"
BACKEND_DIR = MAJOR_PROJECT_DIR / "backend"
MODELS_DIR = MAJOR_PROJECT_DIR / "models"

API_BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

CATEGORY_ICONS = {
    "invoice": "🧾",
    "receipt": "🏷️",
    "form": "📋",
    "letter": "✉️",
    "report": "📊",
}

CATEGORY_DESCRIPTIONS = {
    "invoice": "Commercial billing statement featuring vendor/client details, tabular itemized rows (Quantity, Unit Rate, Total), subtotal boxes, and Net-30 payment terms.",
    "receipt": "Thermal point-of-sale receipt with store header, timestamp, vertical price lists, sales tax calculations, amount due, and bottom barcode.",
    "form": "Structured questionnaire or registration application containing labeled input boxes, status checkboxes ([X]), declaration clauses, and signature lines.",
    "letter": "Formal business correspondence with organizational letterhead, recipient address block, formal salutation, justified body paragraphs, and sign-off.",
    "report": "Executive summary or technical report with dark colored banner header, analytical sub-headings, statistical chart graphics, and pagination footers.",
}


def check_api_health() -> Optional[Dict[str, Any]]:
    """Check connectivity to the FastAPI backend."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def call_api_predict(image_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Submit document image to the FastAPI prediction endpoint without silent fallbacks."""
    try:
        files = {"file": (filename, image_bytes, "image/png")}
        data = {"top_k": 5}
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, data=data, timeout=10.0)
        if resp.status_code == 200:
            result = resp.json()
            result["_source"] = f"REST API ({API_BASE_URL}/predict)"
            return result
        else:
            st.error(f"REST API Error ({resp.status_code}): {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ **Connection Failed**: The FastAPI service at `{API_BASE_URL}` is unreachable.\n\n"
            "To start the REST API service, run:\n"
            "```powershell\n"
            "cd Week-4/major_project\n"
            "uvicorn backend.app:app --host 127.0.0.1 --port 8000\n"
            "```\n"
            "*Alternatively, you may switch to 'Standalone Direct Model' mode in the sidebar.*"
        )
        return None
    except Exception as e:
        st.error(f"Error communicating with REST API: {e}")
        return None


def run_standalone_direct_inference(image_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Explicitly perform direct in-process inference using model_utils when standalone mode is chosen."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    try:
        from model_utils import load_model_and_metadata, predict_document_image
        model, _, idx_to_class, _ = load_model_and_metadata(str(MODELS_DIR))
        image = Image.open(io.BytesIO(image_bytes))
        top_cat, top_conf, top_k_list, latency = predict_document_image(model, idx_to_class, image, top_k=5)
        return {
            "predicted_category": top_cat,
            "confidence": top_conf,
            "confidence_percentage": f"{top_conf * 100:.2f}%",
            "top_k": top_k_list,
            "inference_time_ms": latency,
            "filename": filename,
            "_source": "Standalone Direct Model (In-Process)",
        }
    except Exception as e:
        st.error(f"Direct model execution failed: {e}")
        return None


def main():
    st.title("📄 Document Understanding & Layout Intelligence")
    st.markdown(
        """
        End-to-End Deep Learning system for classifying unstructured business document images into 
        standardized categories: **Invoice**, **Receipt**, **Form**, **Letter**, and **Report**.
        """
    )

    # Sidebar: System Status & API configuration
    with st.sidebar:
        st.header("⚙️ Execution Configuration")
        exec_mode = st.radio(
            "Inference Architecture:",
            ["REST API (Microservice)", "Standalone Direct Model (Offline)"],
            index=0,
            help="Select 'REST API' to query the FastAPI endpoint, or 'Standalone Direct Model' to evaluate the model in-process without network overhead."
        )

        st.markdown("---")
        st.header("🔌 System Status")
        health = check_api_health()
        if health and health.get("status") == "healthy":
            st.success("✅ FastAPI Backend: **Online**")
            st.caption(f"Endpoint: `{API_BASE_URL}`")
            st.markdown(f"**Model:** `{health.get('model_name')}`")
            st.markdown(f"**API Version:** `v{health.get('version')}`")
        else:
            st.warning("⚠️ FastAPI Backend: **Offline**")
            st.caption(f"Expected at: `{API_BASE_URL}`")

        st.markdown("---")
        st.subheader("📚 Supported Categories")
        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            icon = CATEGORY_ICONS.get(cat, "📄")
            with st.expander(f"{icon} {cat.title()}"):
                st.caption(desc)

    # Main content: Selection / Upload
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Document Input")
        tab_sample, tab_upload = st.tabs(["📁 Quick Presets", "📤 Custom Upload"])

        selected_image: Optional[Image.Image] = None
        selected_filename = "document.png"

        with tab_sample:
            st.markdown("Select a sample document from the benchmark archive:")
            preset_cats = ["invoice", "receipt", "form", "letter", "report"]
            preset_cols = st.columns(len(preset_cats))

            for i, cat in enumerate(preset_cats):
                btn = preset_cols[i].button(
                    f"{CATEGORY_ICONS[cat]} {cat.capitalize()}",
                    key=f"btn_{cat}",
                    use_container_width=True,
                )
                if btn:
                    sample_path = DATA_DIR / cat / "doc_000.png"
                    if sample_path.exists():
                        st.session_state["active_doc"] = str(sample_path)
                        st.session_state["doc_name"] = f"sample_{cat}.png"

            active_path_str = st.session_state.get("active_doc", None)
            if active_path_str and os.path.exists(active_path_str):
                selected_image = Image.open(active_path_str)
                selected_filename = st.session_state.get("doc_name", "sample.png")

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Upload a document image (PNG, JPG, TIFF, WebP):",
                type=["png", "jpg", "jpeg", "tiff", "webp"],
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                selected_filename = uploaded_file.name

        if selected_image is not None:
            st.image(selected_image, caption=f"Document Preview: {selected_filename}", use_container_width=True)
            classify_clicked = st.button("🔍 Classify Document Layout", type="primary", use_container_width=True)
        else:
            st.info("Select a preset document above or upload an image to begin classification.")
            classify_clicked = False

    with col_results:
        st.subheader("2. Layout Intelligence & Prediction")

        if classify_clicked and selected_image is not None:
            buf = io.BytesIO()
            selected_image.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            with st.spinner("Analyzing document visual features..."):
                if exec_mode == "REST API (Microservice)":
                    result = call_api_predict(img_bytes, selected_filename)
                else:
                    result = run_standalone_direct_inference(img_bytes, selected_filename)

            if result:
                st.session_state["last_result"] = result

        # Display result
        result = st.session_state.get("last_result", None)
        if result:
            top_cat = result["predicted_category"]
            conf = result["confidence"]
            icon = CATEGORY_ICONS.get(top_cat, "📄")
            source_badge = result.get("_source", "Unknown")

            # Source attribution badge
            st.caption(f"🛡️ **Inference Source:** `{source_badge}`")

            # Metric card
            st.markdown(
                f"""
                <div style="background-color: #0f172a; border: 2px solid #38bdf8; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
                    <div style="color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">Predicted Category</div>
                    <div style="color: #f8fafc; font-size: 32px; font-weight: bold; margin: 4px 0;">{icon} {top_cat.upper()}</div>
                    <div style="color: #38bdf8; font-size: 18px; font-weight: 600;">Confidence: {result['confidence_percentage']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Latency", f"{result['inference_time_ms']} ms")
            m2.metric("Class Confidence", f"{conf * 100:.1f}%")

            # Top-K Chart
            st.markdown("#### Category Probability Distribution")
            top_k = result["top_k"]
            for item in top_k:
                c_name = item["category"]
                c_conf = item["confidence"]
                c_icon = CATEGORY_ICONS.get(c_name, "📄")
                st.write(f"**{c_icon} {c_name.capitalize()}** — `{c_conf * 100:.1f}%`")
                st.progress(float(min(max(c_conf, 0.0), 1.0)))

            # Category Explanation
            st.markdown("#### Layout Feature Explanation")
            st.info(CATEGORY_DESCRIPTIONS.get(top_cat, "Identified based on spatial layout characteristics."))
        else:
            st.write("Ready for analysis. Prediction metrics, top-k probabilities, and layout features will appear here.")


if __name__ == "__main__":
    main()
