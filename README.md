---
title: Chatbot Main
emoji: 👀
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
base_path: /
pinned: false
license: cc-by-nc-nd-4.0
---

# Materials AI Chatbot & Phase Diagram Generator

An AI-powered materials science assistant that provides:

- PDF-based semantic question answering with retrieval-augmented generation
- Phase diagram generation from structured materials datasets
- A FastAPI backend and Panel frontend
- Support for either Hugging Face models or Ollama models

## Architecture

The application is split into two main services. The frontend is a Panel app that provides the chat interface and the phase-diagram workflows. It sends authenticated requests to a FastAPI backend, which handles PDF ingestion, retrieval, model selection, answer generation, and the phase-diagram endpoints.

Under the hood, the backend uses a shared storage area for PDFs, vector stores, logs, materials datasets, and generated outputs. For language models, the runtime can switch between two offline backends: Hugging Face Transformers or a locally running Ollama server. In Docker and HF Spaces deployments, `start.sh` starts both the backend and frontend together; when running from source, `backend.sh` and `frontend.sh` are started separately.

## Features

### PDF chatbot

- PDF ingestion
- sentence-transformer embeddings
- FAISS vector similarity search
- LLM answer generation with either Hugging Face or Ollama

### Phase diagram generator

- Curie / Neel temperature extraction workflows
- interactive visualisation with hvPlot and HoloViews
- script-based and LLM-assisted generation paths

## 3 Modes of Operation

### 1. [Offline] From source - using either Ollama or HF Transformers

#### Prerequisites

- Conda installed
- Python 3.11 available for the `pdfchat` environment
- `git` and `git-lfs` installed
- Access to the Hugging Face dataset repo used for storage sync
- Internet access on first run for dependency, model, and dataset downloads
- If using Ollama: Ollama installed on the host and the selected local model already pulled

Example Ollama model setup:

```bash
ollama pull deepseek-r1:1.5b
```

#### Environment variables

Keep these secrets private and never commit them:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

Set the runtime variables before starting the app:

- `API_KEY`
- `WHICH_PIPELINE` with value `hf` or `ollama`
- `HF_MODEL` if `WHICH_PIPELINE=hf`
- `OLLAMA_MODEL` if `WHICH_PIPELINE=ollama`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434` if using Ollama
- `STORAGE_DIR=./storage`

#### Storage directory

Use:

```bash
export STORAGE_DIR="./storage"
```

`backend.sh` clones the dataset repo into `./storage` if it is not already present, then keeps syncing generated outputs back to that repo periodically.

#### Example setup

Hugging Face Transformers:

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="hf"
export HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export STORAGE_DIR="./storage"
```

Ollama:

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="ollama"
export OLLAMA_MODEL="deepseek-r1:1.5b"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export STORAGE_DIR="./storage"
```

#### Installation instructions

Clone the repository:

```bash
git clone <your-repo-url>
cd Chatbot_main
```

Create the Conda environment expected by the scripts:

```bash
conda create -n pdfchat python=3.11
conda activate pdfchat
```

Initialize Git LFS:

```bash
git lfs install
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

#### Running instructions

Start the backend:

```bash
bash backend.sh
```

This starts FastAPI on [http://127.0.0.1:9000](http://127.0.0.1:9000).

In a second terminal with the same environment variables, start the frontend:

```bash
bash frontend.sh
```

This starts the Panel UI on [http://127.0.0.1:5006](http://127.0.0.1:5006).

#### Operational notes

- `backend.sh` and `frontend.sh` both expect the Conda environment to be named `pdfchat`
- The frontend and backend must use the same `API_KEY`
- `HF_TOKEN` is used for dataset clone and periodic sync
- `HUGGINGFACE_HUB_TOKEN` is needed for authenticated Hugging Face model access
- If `WHICH_PIPELINE=ollama`, the Ollama server must already be reachable at `OLLAMA_BASE_URL`
- On first run, model downloads and vector-store preparation may take time

### 2. [Offline] With Docker - using either Ollama or HF Transformers

#### Prerequisites

- Docker installed and running
- Access to the same Hugging Face credentials used for models and dataset storage
- If using Ollama: Ollama running on the host machine with the selected model already pulled

#### Environment variables

Keep these secrets private and never bake them into the image:

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

Set the runtime variables for the container:

- `API_KEY`
- `WHICH_PIPELINE` with value `hf` or `ollama`
- `HF_MODEL` if `WHICH_PIPELINE=hf`
- `OLLAMA_MODEL` if `WHICH_PIPELINE=ollama`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434` if using Ollama
- `STORAGE_DIR=/app/storage`

#### Storage directory

Use:

```bash
export STORAGE_DIR="/app/storage"
```

Inside the container, `start.sh` uses `/app/storage`. If that directory is empty, it clones the dataset repo there and then syncs generated outputs periodically.

#### Example setup

Hugging Face Transformers:

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="hf"
export HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export STORAGE_DIR="/app/storage"
```

Ollama:

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="ollama"
export OLLAMA_MODEL="deepseek-r1:1.5b"
export OLLAMA_BASE_URL="http://host.docker.internal:11434"
export STORAGE_DIR="/app/storage"
```

#### Installation instructions

Clone the repository:

```bash
git clone <your-repo-url>
cd Chatbot_main
```

Build the image:

```bash
docker build -t materials-chatbot .
```

#### Running instructions

Run the container:

```bash
docker run --rm -it \
  -p 7860:7860 \
  -p 9000:9000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e API_KEY="$API_KEY" \
  -e WHICH_PIPELINE="$WHICH_PIPELINE" \
  -e HF_MODEL="$HF_MODEL" \
  -e OLLAMA_MODEL="$OLLAMA_MODEL" \
  -e OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  -e STORAGE_DIR="/app/storage" \
  materials-chatbot
```

To persist storage outside the container:

```bash
docker run --rm -it \
  -p 7860:7860 \
  -p 9000:9000 \
  -v "$(pwd)/storage:/app/storage" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  -e API_KEY="$API_KEY" \
  -e WHICH_PIPELINE="$WHICH_PIPELINE" \
  -e HF_MODEL="$HF_MODEL" \
  -e OLLAMA_MODEL="$OLLAMA_MODEL" \
  -e OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
  -e STORAGE_DIR="/app/storage" \
  materials-chatbot
```

Container endpoints:

- Frontend: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- Backend API: [http://127.0.0.1:9000](http://127.0.0.1:9000)

#### Operational notes

- `start.sh` starts both FastAPI and Panel inside the container
- `STORAGE_DIR` should stay `/app/storage` in Docker
- If using Ollama on macOS or Windows, `host.docker.internal` should resolve automatically
- If using Ollama on Linux, add `--add-host=host.docker.internal:host-gateway` to `docker run`
- The same `API_KEY` is used internally by the frontend and backend
- The first container start may take time because models and dataset assets may need to be downloaded

### 3. [Online] HF Spaces with Docker - using only the HF Transformers

#### Prerequisites

- A Hugging Face Space configured with `sdk: docker`
- This repository pushed to the Space
- Access to the required Hugging Face credentials
- No Ollama setup is needed for this mode

#### Environment variables

In the Space settings, configure these secrets and variables:

- `HF_TOKEN` as a secret
- `HUGGINGFACE_HUB_TOKEN` as a secret
- `API_KEY`
- `WHICH_PIPELINE=hf`
- `HF_MODEL`
- `STORAGE_DIR=/app/storage`

Do not set Ollama-only variables for this mode.

#### Storage directory

Use:

```bash
STORAGE_DIR=/app/storage
```

The Space runs inside the Docker container, so the app uses the same `/app/storage` path as local Docker mode.

#### Example setup

Configure the following in the Hugging Face Space secrets and variables UI:

```bash
HF_TOKEN=your_hf_dataset_token
HUGGINGFACE_HUB_TOKEN=your_hf_model_token
API_KEY=your_secure_api_key
WHICH_PIPELINE=hf
HF_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
STORAGE_DIR=/app/storage
```

#### Installation instructions

Create or open a Hugging Face Space that uses Docker, then push this repository to it. The included `Dockerfile` installs dependencies, copies the application into `/app`, exposes port `7860`, and starts the app with `start.sh`.

#### Running instructions

No separate manual run command is needed inside the Space. Once the repository is pushed and the Space secrets are configured, Hugging Face builds the Docker image and starts the application automatically.

The frontend is served by Panel on the public Space URL, and the backend runs alongside it inside the same container.

#### Operational notes

- This mode is intended for Hugging Face Transformers only, so keep `WHICH_PIPELINE=hf`
- `start.sh` serves Panel on port `7860`, which matches the Space configuration
- The current startup script allows websocket origin `dsit-tests-chatbot-main.hf.space` and `localhost:7860`
- `HF_TOKEN` is still required because the app clones and syncs the dataset-backed storage repo
- `HUGGINGFACE_HUB_TOKEN` is required for authenticated model access when needed
- Build and startup times may be longer on first deployment because the Space needs to install dependencies and download model assets
