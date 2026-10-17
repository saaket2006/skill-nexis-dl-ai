"""
app.py
======
Main FastAPI application entrypoint for the Document Understanding Capstone.
Manages application lifecycle, model artifact loading, CORS, and routing.
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model_utils import load_model_and_metadata
from routes import router

MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup and cleanly release resources on shutdown."""
    print(f"Loading document classification model from {MODELS_DIR}...")
    try:
        model, class_to_idx, idx_to_class, metadata = load_model_and_metadata(MODELS_DIR)
        app.state.model = model
        app.state.class_to_idx = class_to_idx
        app.state.idx_to_class = idx_to_class
        app.state.class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        app.state.model_name = metadata.get("model_name", "DocumentLayoutClassifier")
        print("Document classifier model and metadata loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        app.state.model = None
        app.state.class_to_idx = None
        app.state.idx_to_class = None
        app.state.class_names = []

    yield

    # Clean shutdown
    app.state.model = None


app = FastAPI(
    title="Document Understanding & Classification API",
    description="Deep Learning REST API for classifying business document images (invoices, receipts, forms, letters, reports).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
