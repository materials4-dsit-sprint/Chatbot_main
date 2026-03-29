#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Optional: show commands as they run
# set -x


cd "$(dirname "$0")"

echo "Activating conda environment: pdfchat"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdfchat

echo "Starting backend server (FastAPI + Uvicorn)..."
uvicorn server:app \
  --host 127.0.0.1 \
  --port 9000

