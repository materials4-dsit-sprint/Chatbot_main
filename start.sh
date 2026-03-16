#!/usr/bin/env bash
set -e

echo "Starting container..."

# Create storage directory if it doesn't exist
mkdir -p /app/storage

echo "Downloading dataset storage..."

# Clone dataset repo if storage is empty
if [ ! "$(ls -A /app/storage)" ]; then
    git clone https://hf:$HF_TOKEN@huggingface.co/datasets/DSIT-TESTS/materials_dataset /tmp/dataset
    cp -r /tmp/dataset/* /app/storage/ || true
fi

echo "Starting backend (FastAPI)..."

uvicorn pdf_chatbot_server:app \
  --host 0.0.0.0 \
  --port 9000 &

echo "Starting frontend (Panel)..."

python -m panel serve frontend_app.py \
  --address 0.0.0.0 \
  --port 7860 \
  --allow-websocket-origin="*" \
  --allow-websocket-origin="localhost:7860"