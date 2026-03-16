#!/usr/bin/env bash

set -e
# set -x

cd "$(dirname "$0")"

echo "Activating conda environment: pdfchat"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdfchat

echo "Starting frontend (Panel)..."
python -m panel serve frontend_app.py \
  --address 127.0.0.1 \
  --port 5006 \
  --allow-websocket-origin="localhost:5006"
