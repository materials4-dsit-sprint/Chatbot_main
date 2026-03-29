---
title: Chatbot Main
emoji: 👀
colorFrom: yellow
colorTo: green
sdk: docker
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

Frontend (Panel) -> FastAPI backend -> retrieval pipeline -> embeddings/vector stores -> LLM runtime (Hugging Face or Ollama)

## Repository Entry Points

This version is started with the provided shell scripts:

- `backend.sh`: activates the `pdfchat` conda environment, clones or reuses the dataset storage repo in `./storage`, starts periodic storage sync, and launches the FastAPI backend on `127.0.0.1:9000`
- `frontend.sh`: activates the `pdfchat` conda environment and launches the Panel frontend on `127.0.0.1:5006`
- `start.sh`: container entrypoint used by Docker; starts the backend on port `9000` and the frontend on port `7860`

## Prerequisites

### Running from source

- Python 3.10+ and Conda
- `git` and `git-lfs`
- Internet access on first run for model and dependency downloads
- Access to the Hugging Face dataset repo used for storage sync

### Running with Docker

- Docker installed and running
- Access to the same required model and dataset credentials

### If using Ollama

- Ollama installed on the host machine
- A local Ollama model pulled in advance, for example:

```bash
ollama pull deepseek-r1:1.5b
```

## Required Environment Variables

The application expects environment variables to be set before startup.

### Secrets that must be kept private

- `HF_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

Do not commit these values to the repository or hard-code them in tracked files.

### Runtime variables

- `API_KEY`: shared secret used by the frontend to authenticate to the backend
- `WHICH_PIPELINE`: selects the LLM backend, either `hf` or `ollama`
- `HF_MODEL`: required when `WHICH_PIPELINE=hf`
- `OLLAMA_MODEL`: required when `WHICH_PIPELINE=ollama`
- `OLLAMA_BASE_URL`: base URL for Ollama; for Docker use `http://host.docker.internal:11434`
- `STORAGE_DIR`: storage root

### Storage directory behavior

- From source: keep `STORAGE_DIR=./storage`
  `backend.sh` clones the dataset repo into `./storage` automatically if it is not already present.
- In Docker: set `STORAGE_DIR=/app/storage`

## Example Environment Setup

Set these variables in your shell before running the app.

### Hugging Face pipeline

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="hf"
export HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export STORAGE_DIR="./storage"
```

### Ollama pipeline from source

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="ollama"
export OLLAMA_MODEL="deepseek-r1:1.5b"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export STORAGE_DIR="./storage"
```

### Ollama pipeline in Docker

```bash
export HF_TOKEN="your_hf_dataset_token"
export HUGGINGFACE_HUB_TOKEN="your_hf_model_token"
export API_KEY="your_secure_api_key"
export WHICH_PIPELINE="ollama"
export OLLAMA_MODEL="deepseek-r1:1.5b"
export OLLAMA_BASE_URL="http://host.docker.internal:11434"
export STORAGE_DIR="/app/storage"
```

## Installation From Source

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Chatbot_main
```

### 2. Create the Conda environment

The startup scripts expect a Conda environment named `pdfchat`.

```bash
conda create -n pdfchat python=3.11
conda activate pdfchat
```

### 3. Install system dependencies

Install `git-lfs` if it is not already available, then initialize it:

```bash
git lfs install
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Run From Source

After exporting the required environment variables:

### 1. Start the backend

```bash
bash backend.sh
```

What `backend.sh` does:

- activates the `pdfchat` Conda environment
- clones the dataset storage repository into `./storage` if needed
- pulls Git LFS data for the storage repo
- starts the FastAPI backend at [http://127.0.0.1:9000](http://127.0.0.1:9000)

### 2. Start the frontend in a second terminal

Use the same shell environment so the frontend sees the same `API_KEY` and pipeline configuration.

```bash
bash frontend.sh
```

The frontend is then available at [http://127.0.0.1:5006](http://127.0.0.1:5006).

## Build And Run With Docker

### 1. Build the image

```bash
docker build -t materials-chatbot .
```

### 2. Run the container

Use `STORAGE_DIR=/app/storage` inside the container. The container entrypoint is `start.sh`, which starts both services.

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

If you want storage data to persist outside the container, mount a host directory:

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

### Notes for Docker and Ollama

- On macOS and Windows, `host.docker.internal` usually resolves automatically
- On Linux, you may need to add:

```bash
--add-host=host.docker.internal:host-gateway
```

### Docker service endpoints

- Frontend: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- Backend API: [http://127.0.0.1:9000](http://127.0.0.1:9000)

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

## Operational Notes

- The backend and frontend must use the same `API_KEY`
- `WHICH_PIPELINE` must match the corresponding model variable you provide
- `HF_TOKEN` is used by the storage clone/sync workflow
- `HUGGINGFACE_HUB_TOKEN` is needed for authenticated Hugging Face model access when using the HF pipeline
- The first run may take time because models, embeddings, and dataset assets may need to be downloaded
