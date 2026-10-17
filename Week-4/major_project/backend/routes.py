"""
routes.py
=========
FastAPI route definitions for Document Layout Classification.
"""

import io
from typing import List
from PIL import Image, UnidentifiedImageError
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request, status

from schemas import PredictionResponse, HealthResponse, TopKItem, ErrorResponse
from model_utils import predict_document_image

router = APIRouter()
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"],
)
async def health_check(request: Request):
    """Check whether the model and API are online and ready for inference."""
    model = getattr(request.app.state, "model", None)
    classes = getattr(request.app.state, "class_names", [])
    model_name = getattr(request.app.state, "model_name", "DocumentLayoutClassifier")

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_name=model_name,
        supported_classes=classes,
        version="1.0.0",
    )


@router.get(
    "/classes",
    response_model=List[str],
    summary="List Supported Document Classes",
    tags=["Metadata"],
)
async def list_classes(request: Request):
    """Return the list of recognized document layout categories."""
    classes = getattr(request.app.state, "class_names", [])
    if not classes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model class metadata is unavailable.",
        )
    return classes


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input or corrupted file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Classify Document Image",
    tags=["Inference"],
)
async def predict_document(
    request: Request,
    file: UploadFile = File(..., description="Image file of the document to classify"),
    top_k: int = Form(5, description="Number of top category probabilities to return (1-5)"),
):
    """
    Accepts an uploaded document image (PNG, JPEG, etc.), preprocesses the layout,
    and returns the predicted category along with ranked top-k probabilities.
    """
    model = getattr(request.app.state, "model", None)
    idx_to_class = getattr(request.app.state, "idx_to_class", None)

    if model is None or idx_to_class is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document classification model is not loaded.",
        )

    # Validate file extension
    filename = file.filename or "unknown.png"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(ALLOWED_EXTENSIONS))}",
        )

    # Read bytes and validate size
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read upload payload: {e}",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum threshold of {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    # Open PIL image
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()  # Verify integrity
        # Reopen because verify() closes the stream
        image = Image.open(io.BytesIO(content))
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File contains invalid or unreadable image data: {e}",
        )

    # Run inference
    safe_top_k = min(max(top_k, 1), 5)
    try:
        top_cat, top_conf, top_k_raw, latency = predict_document_image(
            model=model,
            idx_to_class=idx_to_class,
            image=image,
            top_k=safe_top_k,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {e}",
        )

    top_k_items = [TopKItem(**item) for item in top_k_raw]

    return PredictionResponse(
        predicted_category=top_cat,
        confidence=top_conf,
        confidence_percentage=f"{top_conf * 100.0:.2f}%",
        top_k=top_k_items,
        inference_time_ms=latency,
        filename=filename,
    )
