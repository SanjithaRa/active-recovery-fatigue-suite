# Production Container for Autonomic Fatigue ML Microservice
FROM python:3.13-slim

WORKDIR /app

# Install system build dependencies for C-extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and ML pipeline modules
COPY . .

# Expose microservice REST port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI microservice
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
