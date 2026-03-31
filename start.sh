#!/usr/bin/env bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

: "${HF_TOKEN:?HF_TOKEN must be set with access to the Hugging Face dataset repo}"
: "${DATASET_REPO:?DATASET_REPO must be set, for example 'owner/dataset_name'}"
DATASET_REPO_URL="https://hf:${HF_TOKEN}@huggingface.co/datasets/${DATASET_REPO}"
echo "Using dataset repo: ${DATASET_REPO}"

print_dataset_repo_error() {
    local action="$1"
    echo "[git] ERROR: Failed to ${action} for dataset repo '${DATASET_REPO}'."
    echo "[git] Verify that DATASET_REPO is correct and that HF_TOKEN has access to this dataset repo."
}

run_or_fail() {
    local action="$1"
    shift
    if ! "$@"; then
        print_dataset_repo_error "${action}"
        exit 1
    fi
}

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

    if ! git remote set-url origin "$DATASET_REPO_URL"; then
        print_dataset_repo_error "set git remote"
        cd /app || return 0
        return 1
    fi
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

    if ! git pull --rebase --autostash origin "$current_branch"; then
        print_dataset_repo_error "pull before sync push"
        cd /app || return 0
        return 1
    fi
    if ! git push origin "HEAD:$current_branch"; then
        print_dataset_repo_error "push storage updates"
        cd /app || return 0
        return 1
    fi

    cd /app || return 0
}

start_storage_sync_loop() {
    (
        while true; do
            sleep 900
            echo "Running periodic storage sync..."
            if ! sync_storage_updates; then
                echo "[git] Storage sync failed; will retry on the next sync interval."
            fi
        done
    ) &
}

echo "Starting container..."

# Create storage directory if it doesn't exist
mkdir -p /app/storage

echo "Downloading dataset storage..."

# Clone dataset repo if storage is empty
if [ ! "$(ls -A /app/storage)" ]; then
    run_or_fail "clone dataset repo into /app/storage" git clone "$DATASET_REPO_URL" /app/storage
    cd /app/storage
    run_or_fail "pull git-lfs objects for /app/storage" git lfs pull
    cd /app
fi

start_storage_sync_loop

echo "Starting backend (FastAPI)..."

uvicorn server:app \
  --host 0.0.0.0 \
  --port 9000 &

echo "Starting frontend (Panel)..."

python -m panel serve frontend_app.py \
  --address 0.0.0.0 \
  --port 7860 \
  --allow-websocket-origin="dsit-tests-chatbot-main.hf.space" \
  --allow-websocket-origin="localhost:7860" \
  --session-token-expiration 3600
