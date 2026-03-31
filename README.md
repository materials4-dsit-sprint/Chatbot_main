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

An AI-powered materials science assistant that integrates retrieval-augmented PDF question answering with data-driven phase diagram generation. The system combines semantic search, large language models, and structured materials datasets to enable natural language interaction with scientific literature and materials data.

### Key Features
- PDF-based semantic search and question answering using RAG  
- Embedding-driven retrieval with FAISS and sentence-transformers  
- Flexible LLM support (Hugging Face or Ollama, local or hosted)  
- Phase diagram generation from structured materials datasets  
- Curie / Néel temperature extraction workflows  
- Interactive visualisation using hvPlot and HoloViews  
- Modular FastAPI backend with a Panel-based frontend


## Architecture

The application is split into two main services that work together around a shared storage area. The frontend is a Panel application defined in `frontend_app.py`. It provides the chat UI, file upload flow, model selector, and the two phase-diagram workflows. The frontend does not run the retrieval or model inference logic itself. Instead, it sends authenticated HTTP requests to the FastAPI backend using the shared `API_KEY`, and it reads some static assets such as the logo from `STORAGE_DIR`.

The backend is defined primarily in `server.py` and is responsible for startup-time initialization, retrieval, generation, PDF/CSV ingestion, and phase-diagram endpoints. On startup it reads from `STORAGE_DIR`, discovers the PDF and CSV assets available there, initializes sentence-transformer embeddings, loads or creates FAISS vector stores, and constructs the selected LLM runtime. Chat requests then go through the backend, which retrieves relevant chunks from the loaded vector stores and passes the assembled prompt to either a Hugging Face Transformers model or an Ollama-hosted model. The same backend also exposes the endpoints used by the script-based and LLM-assisted phase-diagram generation code.

Several code paths depend on specific subdirectories under `STORAGE_DIR`. The chatbot expects PDF files under `pdfs/`, PDF FAISS indexes under `pdf_vectorstores/`, materials CSV data under `materials/`, CSV vector stores under `csv_vectorstores/`, and chat logs under `logs/`. Other features also read or write `materials_nollm_log/`, `materials_outputs/`, `logos/`, and `hf_cache/`. In practice, `STORAGE_DIR` is the persistent working area for the whole application: it holds user-ingested documents, derived retrieval indexes, phase-diagram inputs, generated outputs, and runtime caches.

At the moment, the default workflow assumes that `STORAGE_DIR` comes from a Hugging Face dataset repository. Both `backend.sh` for source mode and `start.sh` for Docker/HF Spaces build a Git remote URL using `HF_TOKEN`, then clone that dataset repo into `STORAGE_DIR` if it is missing or empty. They also periodically sync selected generated folders back to the same remote. That behavior can be changed if you want to use your own local storage directory, a different dataset repo, or a different synchronization strategy; the mode-specific sections below explain what to change depending on how you run the app.


## 3 Modes of Operation

Below are three modes of running the application, supporting both offline and online deployments with different model backends.

- [1. [Offline] From source - using either Ollama or HF Transformers](#1-offline-from-source---using-either-ollama-or-hf-transformers)
- [2. [Offline] With Docker - using either Ollama or HF Transformers](#2-offline-with-docker---using-either-ollama-or-hf-transformers)
- [3. [Online] HF Spaces with Docker - using only the HF Transformers](#3-online-hf-spaces-with-docker---using-only-the-hf-transformers)
- [CLI Usage](#cli-usage)


### 1. [Offline] From source - using either Ollama or HF Transformers

#### Prerequisites

- Conda installed
- Python 3.11 available for a Conda environment of your choice, for example `my-env`
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

- `HF_TOKEN`: Hugging Face token used by `backend.sh` to clone the dataset-backed storage repository into `./storage` and to sync generated outputs back to that repo
- `HUGGINGFACE_HUB_TOKEN`: Hugging Face Hub token used when the app needs authenticated access to Hugging Face model artifacts

Set the runtime variables before starting the app:

- `API_KEY`: shared secret used by the frontend when calling the backend API; both services must use the same value
- `WHICH_PIPELINE`: selects which LLM backend to use, either `hf` for Hugging Face Transformers or `ollama` for a local Ollama server
- `HF_MODEL`: the Hugging Face model identifier to load when `WHICH_PIPELINE=hf`
- `OLLAMA_MODEL`: the Ollama model name to call when `WHICH_PIPELINE=ollama`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`: the local Ollama server endpoint used only when `WHICH_PIPELINE=ollama`
- `STORAGE_DIR=./storage`: the local storage root used by the backend for PDFs, vector stores, logs, materials data, and generated outputs

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

#### About `STORAGE_DIR`

In the current setup, `STORAGE_DIR` is not just an empty local folder. `backend.sh` treats it as a clone target for the Hugging Face dataset repo configured in:

```bash
DATASET_REPO_URL="https://hf:${HF_TOKEN}@huggingface.co/datasets/DSIT-TESTS/materials_dataset"
```

The storage directory should contain, or eventually be able to contain, at least these folders:

- `pdfs/`
- `pdf_vectorstores/`
- `materials/`
- `csv_vectorstores/`
- `logs/`
- `materials_nollm_log/`
- `materials_outputs/`
- `logos/`
- `hf_cache/`

If you want to keep using the current Hugging Face dataset repo, you need a valid `HF_TOKEN` that has access to that dataset.

If you want to use your own `STORAGE_DIR`, there are two common options:

- Keep the current clone-and-sync workflow, but change `DATASET_REPO_URL` in [backend.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/backend.sh) to point to your own Hugging Face dataset repo. In that case, `HF_TOKEN` must have access to your repo.
- Stop cloning from Hugging Face and use a purely local folder. In that case, remove or replace the clone/pull/push logic in [backend.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/backend.sh), create/populate `STORAGE_DIR` yourself, and keep the expected folder structure above.

#### Installation instructions

Clone the repository:

```bash
git clone <your-repo-url>
cd Chatbot_main
```

Create a Conda environment with any name you prefer, for example `my-env`:

```bash
conda create -n my-env python=3.11
conda activate my-env
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

- Choose any Conda environment name you prefer, but make sure the activation lines in `backend.sh` and `frontend.sh` match that name before running the scripts
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

- `HF_TOKEN`: Hugging Face token passed into the container so `start.sh` can clone the dataset-backed storage repository into `/app/storage` and sync generated outputs back to it
- `HUGGINGFACE_HUB_TOKEN`: Hugging Face Hub token used inside the container for authenticated access to Hugging Face model artifacts

Set the runtime variables for the container:

- `API_KEY`: shared secret used by the frontend process to authenticate requests to the backend process running in the same container
- `WHICH_PIPELINE`: selects whether the container uses `hf` or `ollama` as the LLM backend
- `HF_MODEL`: the Hugging Face model identifier to load when `WHICH_PIPELINE=hf`
- `OLLAMA_MODEL`: the Ollama model name to call when `WHICH_PIPELINE=ollama`
- `OLLAMA_BASE_URL=http://host.docker.internal:11434`: the Ollama endpoint reachable from inside the container when the model is hosted on the machine running Docker
- `STORAGE_DIR=/app/storage`: the in-container storage root used for PDFs, vector stores, logs, materials data, and generated outputs

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

#### About `STORAGE_DIR`

In Docker mode, `start.sh` currently assumes that `/app/storage` is backed by the Hugging Face dataset repo configured in:

```bash
DATASET_REPO_URL="https://hf:$HF_TOKEN@huggingface.co/datasets/DSIT-TESTS/materials_dataset"
```

If `/app/storage` is empty, `start.sh` clones that repo there automatically and then syncs selected generated outputs back to it.

The storage directory should contain, or be able to contain, these folders:

- `pdfs/`
- `pdf_vectorstores/`
- `materials/`
- `csv_vectorstores/`
- `logs/`
- `materials_nollm_log/`
- `materials_outputs/`
- `logos/`
- `hf_cache/`

If you want to keep the current Hugging Face-backed storage behavior, `HF_TOKEN` must be able to access that dataset repo.

If you want to use your own `STORAGE_DIR` in Docker, update the storage logic in [start.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/start.sh):

- To use your own Hugging Face dataset repo, change `DATASET_REPO_URL` to your repo and pass an `HF_TOKEN` that can access it.
- To use a purely local or mounted storage volume, remove or replace the clone/push logic in [start.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/start.sh), mount or create `/app/storage` yourself, and make sure it has the expected folder structure above.

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

- `HF_TOKEN` as a secret: required so the container running in the Space can clone the dataset-backed storage repository and sync generated outputs
- `HUGGINGFACE_HUB_TOKEN` as a secret: required for authenticated access to Hugging Face model artifacts when the selected model needs it
- `API_KEY`: shared secret used by the Panel frontend in the Space to authenticate requests to the FastAPI backend in the same container
- `WHICH_PIPELINE=hf`: required to force the app onto the Hugging Face Transformers path, since Ollama is not used in this deployment mode
- `HF_MODEL`: the Hugging Face model identifier that the Space should load for inference
- `STORAGE_DIR=/app/storage`: the storage root inside the Docker container used by the Space deployment

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

#### About `STORAGE_DIR`

In HF Spaces Docker mode, the app still uses the same container startup script, so `/app/storage` is currently handled by [start.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/start.sh). That means the Space expects `STORAGE_DIR` to be populated by cloning the configured Hugging Face dataset repo when the directory is empty.

The storage directory should contain, or be able to contain, these folders:

- `pdfs/`
- `pdf_vectorstores/`
- `materials/`
- `csv_vectorstores/`
- `logs/`
- `materials_nollm_log/`
- `materials_outputs/`
- `logos/`
- `hf_cache/`

If you keep the current behavior, the `HF_TOKEN` secret in the Space must have permission to access the configured dataset repo.

If you want to point the Space at your own storage source, edit [start.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/start.sh):

- To use your own Hugging Face dataset repo, change `DATASET_REPO_URL` and provide an `HF_TOKEN` secret that can access that repo.
- To use a different storage bootstrapping approach, replace the clone/sync logic in [start.sh](/Users/kulkarni/Library/CloudStorage/OneDrive-UniversityofCambridge/ChatBot%20Project/Projects/HF/Chatbot_main/start.sh) and ensure `/app/storage` is created with the expected folder layout before the backend starts.

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

## CLI Usage

Several modules in this repository are primarily intended to be imported by the backend or frontend, but the codebase currently exposes two main Python command-line entry points with explicit CLI argument parsing:

- `server.py`
- `chatbot.py`

These CLIs are useful if you want to start the backend directly from Python or run a minimal terminal-based PDF chat workflow without using the Panel frontend.

### When CLI can be used

- `[Offline] From source`: supported; this is the most natural mode for direct CLI usage
- `[Offline] With Docker`: possible, but only by running commands inside the container or by overriding the container entry command; it is not the default way this repo is run in Docker
- `[Online] HF Spaces with Docker`: not intended for CLI usage; HF Spaces builds and runs the container through `start.sh`, and users do not interact with it through these Python CLIs

### `server.py`

Use `server.py` to initialize the retrieval stack from `STORAGE_DIR`, load the selected model pipeline, and start the FastAPI backend directly from the command line.

Example:

```bash
python server.py \
  --pdfs-dir ./storage/pdfs \
  --vs-dir ./storage/pdf_vectorstores \
  --host 127.0.0.1 \
  --port 9000
```

Supported arguments:

- `--pdfs-dir`: directory containing the PDF files to index and serve for retrieval
- `--vs-dir`: directory where PDF FAISS vector stores are stored or created
- `--hf-model`: Hugging Face model identifier to use when `WHICH_PIPELINE=hf`
- `--ollama-model`: Ollama model name to use when `WHICH_PIPELINE=ollama`
- `--sent-model`: sentence-transformer embedding model name
- `--reindex`: force rebuilding the vector stores instead of reusing existing ones
- `--host`: host interface for the FastAPI server
- `--port`: port for the FastAPI server

What it can be used for:

- starting the backend without `backend.sh`
- testing custom PDF or vector-store directories
- forcing a rebuild of retrieval indexes with `--reindex`
- running the API server for local development or debugging

### `chatbot.py`

Use `chatbot.py` for a lightweight terminal-based chatbot flow around a single PDF. It validates the PDF path, uploads or copies the PDF into the managed PDF directory, creates or loads a vector store, initializes the selected LLM backend, and starts an interactive REPL in the terminal.

Example:

```bash
python chatbot.py /path/to/paper.pdf \
  --pdfs-dir ./storage/pdfs \
  --vs-dir ./storage/pdf_vectorstores \
  -k 30
```

Supported arguments:

- `pdf`: path to the PDF file to ingest for the session
- `--pdfs-dir`: directory where the managed PDF copy is stored
- `--vs-dir`: directory where the PDF vector store is stored or created
- `--hf-model`: Hugging Face model identifier to use when `WHICH_PIPELINE=hf`
- `--ollama-model`: Ollama model name to use when `WHICH_PIPELINE=ollama`
- `--sent-model`: sentence-transformer embedding model name
- `--reindex`: force rebuilding the vector store for the PDF
- `-k`: number of retrieved chunks to use when answering each question

What it can be used for:

- quickly testing retrieval and generation on a single PDF
- debugging model selection without starting the full frontend
- rebuilding a PDF vector store for one document
- running a local terminal REPL instead of the web UI

### Notes

- These CLIs still rely on the same environment variables described in the operation-mode sections above, especially `WHICH_PIPELINE`, `HF_MODEL` or `OLLAMA_MODEL`, `HUGGINGFACE_HUB_TOKEN`, and `STORAGE_DIR`
- For normal local development, CLI usage is best suited to the from-source workflow rather than Docker or HF Spaces
- In offline Docker mode, you can use the CLI only if you explicitly execute `python server.py ...` or `python chatbot.py ...` inside the container environment
- In HF Spaces mode, treat the app as a deployed web service rather than a CLI-driven application
- If your chosen storage layout differs from the current Hugging Face dataset-backed setup, update the relevant directories and startup logic as described in the `STORAGE_DIR` subsections above
- Other Python files in the repository are currently used mainly as imported modules or backend helpers rather than standalone CLI entry points
