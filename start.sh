#!/usr/bin/env bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

DATASET_REPO_URL="https://hf:$HF_TOKEN@huggingface.co/datasets/DSIT-TESTS/materials_dataset"

sync_storage_updates() {
    local sync_paths=()

    if [ ! -d /app/storage/.git ]; then
        echo "Skipping storage sync: /app/storage is not a git repo yet."
        return 0
    fi

    cd /app/storage || return 0

    if [ -e /app/storage/logs ]; then
        sync_paths+=("logs")
    fi
    if [ -e /app/storage/materials_nollm_log ]; then
        sync_paths+=("materials_nollm_log")
    fi
    if [ -e /app/storage/materials_outputs ]; then
        sync_paths+=("materials_outputs")
    fi

    if [ ${#sync_paths[@]} -eq 0 ]; then
        echo "No storage sync paths exist yet."
        cd /app || return 0
        return 0
    fi

    git remote set-url origin "$DATASET_REPO_URL" || true
    git config user.name "Chatbot Container" || true
    git config user.email "chatbot-container@local" || true

    git add -A "${sync_paths[@]}"

    if git diff --cached --quiet -- "${sync_paths[@]}"; then
        echo "No storage updates to push."
        cd /app || return 0
        return 0
    fi

    git commit -m "Sync generated storage updates ($(date -u +"%Y-%m-%dT%H:%M:%SZ"))" || true

    current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
    if [ "$current_branch" = "HEAD" ] || [ -z "$current_branch" ]; then
        current_branch="main"
    fi

    git pull --rebase --autostash origin "$current_branch" || true
    git push origin "HEAD:$current_branch" || true

    cd /app || return 0
}

start_storage_sync_loop() {
    (
        while true; do
            sleep 900
            echo "Running periodic storage sync..."
            sync_storage_updates
        done
    ) &
}

echo "Starting container..."

# Create storage directory if it doesn't exist
mkdir -p /app/storage

echo "Downloading dataset storage..."

# Clone dataset repo if storage is empty
if [ ! "$(ls -A /app/storage)" ]; then
    git clone "$DATASET_REPO_URL" /app/storage
    cd /app/storage
    git lfs pull
    cd /app
fi

start_storage_sync_loop

echo "Starting backend (FastAPI)..."

uvicorn pdf_chatbot_server:app \
  --host 0.0.0.0 \
  --port 9000 &

echo "Starting frontend (Panel)..."

python -m panel serve frontend_app.py \
  --address 0.0.0.0 \
  --port 7860 \
  --allow-websocket-origin="*" \
  --allow-websocket-origin="localhost:7860" \
  --session-token-expiration 3600
