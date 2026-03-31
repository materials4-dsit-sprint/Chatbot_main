#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Optional: show commands as they run
# set -x


cd "$(dirname "$0")"

DATASET_REPO_DIR="./storage"
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

        if ! git remote set-url origin "${DATASET_REPO_URL}"; then
            print_dataset_repo_error "set git remote"
            exit 1
        fi
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

        if ! git pull --rebase --autostash origin "${current_branch}"; then
            print_dataset_repo_error "pull before sync push"
            exit 1
        fi
        if ! git push origin "HEAD:${current_branch}"; then
            print_dataset_repo_error "push storage updates"
            exit 1
        fi
    )
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

echo "Activating conda environment: pdfchat"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pdfchat

if [ ! -d "${DATASET_REPO_DIR}/.git" ]; then
    echo "Cloning dataset storage into ${DATASET_REPO_DIR}..."
    rm -rf "${DATASET_REPO_DIR}"
    run_or_fail "clone dataset repo into ${DATASET_REPO_DIR}" git clone "${DATASET_REPO_URL}" "${DATASET_REPO_DIR}"
    (
        cd "${DATASET_REPO_DIR}" || exit 0
        run_or_fail "pull git-lfs objects for ${DATASET_REPO_DIR}" git lfs pull
    )
    echo "[git] Dataset repo cloned successfully into ${DATASET_REPO_DIR}"
else
    echo "[git] Dataset repo already present at ${DATASET_REPO_DIR}; pulling latest changes"
    (
        cd "${DATASET_REPO_DIR}" || exit 0
        run_or_fail "set git remote" git remote set-url origin "${DATASET_REPO_URL}"
        current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
        if [ "${current_branch}" = "HEAD" ] || [ -z "${current_branch}" ]; then
            current_branch="main"
        fi
        run_or_fail "pull latest dataset repo changes" git pull --rebase --autostash origin "${current_branch}"
        run_or_fail "pull git-lfs objects for ${DATASET_REPO_DIR}" git lfs pull
    )
    echo "[git] Dataset repo updated successfully at ${DATASET_REPO_DIR}"
fi

start_storage_sync_loop

echo "Starting backend server (FastAPI + Uvicorn)..."
uvicorn server:app \
  --host 127.0.0.1 \
  --port 9000
