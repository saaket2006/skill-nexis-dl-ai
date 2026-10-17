# Multi-stage or lightweight Python image for Document Classification Service
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (curl for healthchecks, libgl for image processing if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy application source code and trained model artifacts
COPY backend /app/backend
COPY frontend /app/frontend
COPY models /app/models
COPY training /app/training
COPY data /app/data

# Ensure permissions
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Healthcheck for FastAPI API
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts the FastAPI REST service
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
