# ============================================================
# FraudGuard 360 — Member 4 Model Service
# Pinned demo model: fraudguard360-detector@6 (logreg-1.0-v6)
# Feature schema: 1.0 (12 canonical Member 3 features)
# POLICY_VERSION: v1
# ============================================================

FROM python:3.12-slim

# Metadata labels
LABEL org.opencontainers.image.title="FraudGuard 360 Model Service" \
      org.opencontainers.image.description="M4 ML Inference & SHAP Explanation Service" \
      org.opencontainers.image.version="1.0.0" \
      fraudguard.model.name="fraudguard360-detector" \
      fraudguard.model.version="6" \
      fraudguard.feature.schema="1.0" \
      fraudguard.policy.version="v1"

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM / OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py ./
COPY model_loader.py ./
COPY explain.py ./
COPY conftest.py ./
COPY train.py ./
COPY dummy_data.py ./

# Copy pinned model artifact (fraudguard360-detector@6)
COPY models/ ./models/

# Copy repository-contained training dataset (12-feature schema)
COPY dummy_data/ ./dummy_data/

# Copy test suite
COPY tests/ ./tests/

# Expose service port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/health')" || exit 1

# Default: run uvicorn on port 8001
# DATA_SOURCE=dummy by default (deterministic, reproduces fraudguard360-detector@6)
ENV DATA_SOURCE=dummy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
