---
title: Chatbot Main
emoji: 👀
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
license: cc-by-nc-nd-4.0
---

# 📘 Materials AI Chatbot & Phase Diagram Generator

An AI-powered materials science assistant that:

- 📄 Performs PDF-based semantic question answering (RAG pipeline)
- 📊 Generates phase diagrams from structured datasets
- 🤖 Integrates LLM reasoning with experimental temperature data
- 🌐 Provides an interactive Panel-based web interface

---

# 🏗️ Architecture Overview

Frontend (Panel + hvPlot) ⬇ FastAPI Backend ⬇ LangChain RAG Pipeline ⬇ Sentence-Transformers + FAISS ⬇ LLM (DeepSeek / local model)

---

# 📦 Installation Guide

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/materials-chatbot.git
cd materials-chatbot
```

---

### 2️⃣ Create Conda Environment (Recommended)

```bash
conda create -n pdfchat python=3.10
conda activate pdfchat
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variable Setup (Required)

This project requires an API key for frontend–backend authentication.

### Add API\_KEY to your \~/.bashrc

```bash
nano ~/.bashrc
```

Add:

```bash
export API_KEY="your_secure_key_here"
```

Reload:

```bash
source ~/.bashrc
```

Verify:

```bash
echo $API_KEY
```

Optional pipeline selection:

```bash
export WHICH_PIPELINE="hf"      # or "ollama"
export HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
export OLLAMA_MODEL="deepseek-r1:1.5b"
```

---

# 🤖 Installing DeepSeek Models (Local LLM)

This project can run using DeepSeek 8B or 1.5B locally.

### 1️⃣ Install Ollama

Linux/macOS:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

---

### 2️⃣ Pull DeepSeek Models

**DeepSeek 8B (Higher Quality)**

```bash
ollama pull deepseek-r1:8b
```

**DeepSeek 1.5B (Lightweight / Lower RAM)**

```bash
ollama pull deepseek-r1:1.5b
```

---

### 3️⃣ Test Model

```bash
ollama run deepseek-r1:8b
```

(or replace with `1.5b`)

---

# 🚀 Running the Application

---

### ▶ Start Backend

```bash
bash backend.sh
```

This script:

- Activates conda environment `pdfchat`
- Starts FastAPI server using Uvicorn
- Runs at [http://127.0.0.1:9000](http://127.0.0.1:9000)

---

### ▶ Start Frontend

Open a new terminal and run:

```bash
bash frontend.sh
```

This script:

- Activates conda environment `pdfchat`
- Launches Panel app
- Runs at [http://127.0.0.1:5006](http://127.0.0.1:5006)


---

# 📊 Features

### 🔹 PDF Chatbot (RAG Pipeline)

- PDF ingestion
- Recursive text splitting
- Embedding generation (Sentence-Transformers)
- FAISS vector similarity search
- LLM-based answer synthesis

### 🔹 Phase Diagram Generator

- Extracts Curie / Néel temperatures
- Supports A(1-x)B(x)C compounds
- Interactive hvPlot visualisation
- Optional LLM-assisted interpretation

---

# ⚠️ Notes

- First run requires internet to download embedding models.
- DeepSeek 8B requires significantly more RAM than 1.5B.
- For GPU systems, ensure proper CUDA configuration.
