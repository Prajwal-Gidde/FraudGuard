FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/apps/risk_service

# Expose the service port
EXPOSE 8000

# Set Python path so `apps.risk_service` is importable
ENV PYTHONPATH=/app

CMD ["uvicorn", "apps.risk_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
