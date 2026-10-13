FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy dataset, source code, and scripts
COPY data/ ./data/
COPY src/ ./src/

# Retrain the model inside the container's exact Python/scikit-learn environment
RUN python src/data_loader.py && \
    python src/train_lr.py && \
    mkdir -p models && \
    cp models/Logistic_Regression.joblib models/best_model.joblib

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
