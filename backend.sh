#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Optional: show commands as they run
# set -x


cd "$(dirname "$0")"

DATASET_REPO_DIR="./storage"
DATASET_REPO_URL="https://hf:${HF_TOKEN}@huggingface.co/datasets/DSIT-TESTS/materials_dataset"

sync_storage_updates() {
    local sync_paths=()

    if [ ! -d "${DATASET_REPO_DIR}/.git" ]; then
        echo "Skipping storage sync: ${DATASET_REPO_DIR} is not a git repo yet."
        return 0
    fi

    if [ -e "./storage/logs" ]; then
        sync_paths+=("logs")
    fi
    if [ -e "./storage/materials_nollm_log" ]; then
        sync_paths+=("materials_nollm_log")
    fi
    if [ -e "./storage/materials_outputs" ]; then
        sync_paths+=("materials_outputs")
    fi

    if [ ${#sync_paths[@]} -eq 0 ]; then
        echo "No storage sync paths exist yet."
        return 0
    fi

    for path in "${sync_paths[@]}"; do
        mkdir -p "${DATASET_REPO_DIR}/${path}"
        rsync -a --delete "./storage/${path}/" "${DATASET_REPO_DIR}/${path}/"
    done

    (
        cd "${DATASET_REPO_DIR}" || exit 0

        git remote set-url origin "${DATASET_REPO_URL}" || true
        git config user.name "Chatbot Local Backend" || true
        git config user.email "chatbot-local@local" || true

        git add -A "${sync_paths[@]}"

        if git diff --cached --quiet -- "${sync_paths[@]}"; then
            echo "No storage updates to push."
            exit 0
        fi

        git commit -m "Sync generated storage updates ($(date -u +"%Y-%m-%dT%H:%M:%SZ"))" || true

        current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
        if [ "${current_branch}" = "HEAD" ] || [ -z "${current_branch}" ]; then
            current_branch="main"
        fi

        git pull --rebase --autostash origin "${current_branch}" || true
        git push origin "HEAD:${current_branch}" || true
    )
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

echo "Activating conda environment: pdfchat"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdfchat

if [ ! -d "${DATASET_REPO_DIR}/.git" ]; then
    echo "Cloning dataset storage into ${DATASET_REPO_DIR}..."
    rm -rf "${DATASET_REPO_DIR}"
    git clone "${DATASET_REPO_URL}" "${DATASET_REPO_DIR}"
    (
        cd "${DATASET_REPO_DIR}" || exit 0
        git lfs pull || true
    )
    echo "[git] Dataset repo cloned successfully into ${DATASET_REPO_DIR}"
else
    echo "[git] Dataset repo already present at ${DATASET_REPO_DIR}; pulling latest changes"
    (
        cd "${DATASET_REPO_DIR}" || exit 0
        git remote set-url origin "${DATASET_REPO_URL}" || true
        current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
        if [ "${current_branch}" = "HEAD" ] || [ -z "${current_branch}" ]; then
            current_branch="main"
        fi
        git pull --rebase --autostash origin "${current_branch}"
        git lfs pull || true
    )
    echo "[git] Dataset repo updated successfully at ${DATASET_REPO_DIR}"
fi

start_storage_sync_loop

echo "Starting backend server (FastAPI + Uvicorn)..."
uvicorn server:app \
  --host 127.0.0.1 \
  --port 9000
