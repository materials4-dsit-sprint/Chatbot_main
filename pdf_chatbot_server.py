#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pdf_chatbot_server.py
"""
Load embeddings + multiple vectorstores + Ollama model from PDFs in ./data/pdfs, then serve /generate.
Run with uvicorn (recommended):
    uvicorn pdf_chatbot_server:app --host 127.0.0.1 --port 9000
Or run directly:
    python pdf_chatbot_server.py --pdfs-dir ./pdfs --vs-dir ~/pdf_vectorstores
"""

import argparse
import glob
import os
import sys
from math import ceil
from typing import Literal
from fastapi import FastAPI, HTTPException, Header, Response
from pydantic import BaseModel
import uvicorn
from starlette.concurrency import run_in_threadpool
from pdf_chatbot import build_prompt, invoke_llm_and_get_text
from embeddings import get_embeddings_provider
import core
# from langchain_ollama.llms import OllamaLLM
from transformers import pipeline
import numpy as np
import pandas as pd
import json

# Defaults
PDFS_DIR_DEFAULT = os.path.join("/app/storage", "pdfs")
VS_DIR_DEFAULT = os.path.join("/app/storage", "pdf_vectorstores")
# DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"
DEFAULT_HF_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_SENT_MODEL = "all-MiniLM-L6-v2"
ALLOWED_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
]
app = FastAPI()

# import the router
from materials_phase import router as materials_router
app.include_router(materials_router, prefix="")


class GenReq(BaseModel):
    question: str
    k: int | None = 30
    log: bool | None = True
    model: str | None = None
    context_source: Literal["pdfs", "csvs"] = "pdfs"

#get API_KEY
API_KEY = os.environ.get("API_KEY")

# Globals
_DBS: list = []   # loaded vectorstores (one per saved VS)
_PDF_DBS: list = []
_CSV_DBS: list = []
_llm = None
# --- LLM cache for model switching ---
# _LLM_CACHE: dict[str, OllamaLLM] = {}  # this was required for ollama
_LLM_CACHE: dict[str, object] = {}

#------------------------- simple JSONL logger ------------------------------
import uuid, datetime
LOG_PATH = os.path.join("/app/storage", "logs", "chat_logs.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def append_chat_log(entry: dict):
    """Append one JSON object as a line to LOG_PATH (safe append)."""
    try:
        entry.setdefault("ts", datetime.datetime.utcnow().isoformat() + "Z")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # non-fatal: print to stderr so ops can see issues
        print("Chat log write failed:", e, file=sys.stderr)
#----------------------------------------------------------------------------


@app.post("/generate")
async def generate(req: GenReq, authorization: str | None = Header(None)):
    # auth
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _DBS or _llm is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    def work():
        k_used = req.k or 30
        selected_dbs = get_dbs_for_context_source(req.context_source)
        if not selected_dbs:
            source_label = "PDF" if req.context_source == "pdfs" else "CSV"
            return {"text": f"No indexed {source_label} sources are available."}

        docs = combined_retrieve(selected_dbs, req.question, k_used)
        if not docs:
            return {"text": "No relevant documents found."}
    
        rid = str(uuid.uuid4())
    
        # --- Conditional retrieval logging ---
        if req.log:
            retrieved_summary = []
            for d in docs:
                md = getattr(d, "metadata", {}) or {}
                retrieved_summary.append({
                    "source_type": md.get("source_type", "pdf"),
                    "id": md.get("row_id") or md.get("source") or md.get("filename", None),
                    "snippet": (getattr(d, "page_content", "")[:400])
                })
    
            append_chat_log({
                "event": "retrieval",
                "request_id": rid,
                "question": req.question,
                "context_source": req.context_source,
                "k_requested": k_used,
                "retrieved_count": len(retrieved_summary),
                "retrieved": retrieved_summary
            })
    
        prompt = build_prompt(req.question, docs)
    
        # --- Select LLM model (cached) ---
        # selected_model = req.model or DEFAULT_OLLAMA_MODEL
        selected_model = req.model or DEFAULT_HF_MODEL
        
        # Validate against allowlist if present
        try:
            allowed = ALLOWED_MODELS  # expects ALLOWED_MODELS defined near defaults
        except NameError:
            # allowed = [DEFAULT_OLLAMA_MODEL]
            allowed = [DEFAULT_HF_MODEL]
    
        if selected_model not in allowed:
            raise HTTPException(status_code=400, detail="Invalid model selection")
    
        # Get from cache or create and cache
        if selected_model not in _LLM_CACHE:
            print(f"Initializing new LLM instance: {selected_model}", file=sys.stderr)
            # _LLM_CACHE[selected_model] = OllamaLLM(model=selected_model)
            _LLM_CACHE[selected_model] = pipeline("text-generation", model=selected_model, device_map="auto")
    
        llm_instance = _LLM_CACHE[selected_model]
    
        # Invoke the chosen LLM
        text = invoke_llm_and_get_text(llm_instance, prompt)
    
        # --- Conditional answer logging ---
        if req.log:
            append_chat_log({
                "event": "answer",
                "request_id": rid,
                "question": req.question,
                "context_source": req.context_source,
                "k_used": k_used,
                "llm_answer": text,
                "context_length_chars": len(prompt),
                "model_used": selected_model,
            })
    
        return {"text": text}
    result = await run_in_threadpool(work)
    return result

# ----------------- multi-vectorstore utilities -----------------

def load_vectorstores_from_dir(vs_dir: str, embeddings):
    """
    Best-effort: scan vs_dir for candidate vectorstore folders/files and try to load them.
    Returns a list of loaded DB objects (LangChain-like or core's DB objects).
    """
    vs_dir = os.path.abspath(os.path.expanduser(vs_dir))
    loaded = []

    if not os.path.isdir(vs_dir):
        print(f"Vectorstore dir {vs_dir} not found; creating.", file=sys.stderr)
        os.makedirs(vs_dir, exist_ok=True)
        return loaded

    candidates = sorted([os.path.join(vs_dir, p) for p in os.listdir(vs_dir)])
    for cand in candidates:
        if not os.path.exists(cand):
            continue

        # Try core.load_vector_store first (if present)
        try:
            if hasattr(core, "load_vector_store"):
                db = core.load_vector_store(cand, embeddings)
                setattr(db, "_source_path", cand)
                print(f"Loaded vectorstore via core.load_vector_store: {cand}", file=sys.stderr)
                loaded.append(db)
                continue
        except Exception as e:
            print(f"core.load_vector_store failed for {cand}: {e}", file=sys.stderr)

        # Try core.create_or_load_vector_store as a fallback
        try:
            db, store_dir = core.create_or_load_vector_store(cand, vs_dir, embeddings, reindex=False)
            setattr(db, "_source_path", cand)
            print(f"Loaded vectorstore via core.create_or_load_vector_store fallback: {cand}", file=sys.stderr)
            loaded.append(db)
            continue
        except Exception:
            pass

        # Try LangChain FAISS local load (best-effort)
        try:
            # from langchain.vectorstores import FAISS
            # heuristics: FAISS saved folder or .index file
            if os.path.isdir(cand) or cand.endswith(".index"):
                try:
                    from langchain_community.vectorstores import FAISS as _FAISS

                    db = _FAISS.load_local(cand, embeddings, allow_dangerous_deserialization=True)
                    setattr(db, "_source_path", cand)
                    print(f"Loaded FAISS from {cand}", file=sys.stderr)
                    loaded.append(db)
                    continue
                except Exception:
                    pass
        except Exception:
            pass

        print(f"Skipping candidate (not a supported vectorstore): {cand}", file=sys.stderr)

    return loaded

def infer_db_source_type(db, candidate_path: str) -> str:
    """
    Best-effort DB classification so we can keep PDF and CSV retrieval separate.
    """
    base = os.path.basename(candidate_path).lower()
    if "csv" in base:
        return "csv"

    try:
        if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
            docs = list(db.docstore._dict.values())
            if docs:
                metadata = getattr(docs[0], "metadata", {}) or {}
                if metadata.get("source_type") == "csv":
                    return "csv"
    except Exception:
        pass

    return "pdf"

def get_dbs_for_context_source(context_source: str):
    if context_source == "csvs":
        return _CSV_DBS
    return _PDF_DBS

def combined_retrieve(dbs, query: str, k_total: int):
    """
    Query multiple DBs and merge results. Returns up to k_total documents.
    Each db is queried for k_per_db = ceil(k_total / len(dbs)).
    Deduplicates by page content (or string) and returns list of doc objects.
    """
    if not dbs:
        return []

    # request more per DB to allow for dedup/uneven coverage
    overfetch_factor = 2   # try 2x; increase if needed
    k_per_db = max(1, ceil((k_total * overfetch_factor) / len(dbs)))
    collected = []

    for db in dbs:
        docs = []
        try:
            # try core.retrieve_docs if it accepts (db, query, k)
            if hasattr(core, "retrieve_docs"):
                docs = core.retrieve_docs(db, query, k=k_per_db) or []
            elif hasattr(db, "similarity_search_with_score"):
                docs = [d for d, score in db.similarity_search_with_score(query, k=k_per_db)]
            elif hasattr(db, "similarity_search"):
                docs = db.similarity_search(query, k=k_per_db)
            else:
                docs = []
        except Exception as e:
            print(f"Warning: retrieval from a DB failed: {e}", file=sys.stderr)
            docs = []

        for d in docs:
            text = getattr(d, "page_content", None)
            if text is None:
                try:
                    text = str(d)
                except Exception:
                    text = ""
            collected.append((text, d))

    # deduplicate by content
    seen = set()
    deduped = []
    for text, doc in collected:
        key = text.strip()[:4096]
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
        if len(deduped) >= k_total:
            break
    print(f"[retrieve] requested k_total={k_total}, collected={len(collected)}, deduped={len(deduped)}", file=sys.stderr)
    return deduped[:k_total]

# ----------------- initialization -----------------

# def init_services_from_pdfs(pdfs_dir: str, vs_dir: str, sent_model: str, ollama_model: str, reindex: bool):
def init_services_from_pdfs(pdfs_dir: str, vs_dir: str, sent_model: str, hf_model: str, reindex: bool):
    """
    Initialize embeddings, load/create vectorstores (one per saved VS or per PDF),
    and initialize the Ollama LLM. Populates global _DBS and _llm.
    """
    global _DBS, _PDF_DBS, _CSV_DBS, _llm

    # find PDFs in pdfs_dir
    pdfs_dir = os.path.abspath(os.path.expanduser(pdfs_dir))
    print(f"Looking for PDFs in: {pdfs_dir}", file=sys.stderr)
    pdf_paths = sorted(glob.glob(os.path.join(pdfs_dir, "*.pdf")))
    
    if not pdf_paths:
        # If no PDFs, check whether a CSV dataset exists and proceed in CSV-only mode
        csv_path_env = os.environ.get("MATERIALS_CSV", None)
        if not csv_path_env:
            csv_path_env = os.path.join("/app/storage", "materials", "materials.csv")
        else:
            csv_path_env = os.path.expanduser(csv_path_env)
    
        if os.path.exists(csv_path_env):
            print(f"No PDF files found in {pdfs_dir}, but found CSV at {csv_path_env}. Proceeding in CSV-only mode.", file=sys.stderr)
            pdf_paths = []  # keep empty, but don't abort — CSV indexing will run later
        else:
            print(f"No PDF files found in {pdfs_dir}. Put PDFs there (e.g. ./data/pdfs) or provide a CSV at {csv_path_env}.", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s):", file=sys.stderr)
    for p in pdf_paths:
        print("  -", p, file=sys.stderr)

    # embeddings
    try:
        embeddings = get_embeddings_provider(model_name=sent_model)
    except Exception as e:
        print("Failed to initialize sentence-transformers embeddings:", e, file=sys.stderr)
        sys.exit(1)

    # Ollama
    try:
        # _llm = OllamaLLM(model=ollama_model)
        _llm = pipeline("text-generation", model=hf_model, device_map="auto")
    except Exception as e:
        print("Failed to initialize Ollama LLM:", e, file=sys.stderr)
        print("Make sure Ollama is running and the model exists.", file=sys.stderr)
        sys.exit(1)

    # Try loading existing vectorstores from vs_dir
    _DBS = load_vectorstores_from_dir(vs_dir, embeddings)
    _PDF_DBS = []
    _CSV_DBS = []
    for db in _DBS:
        source_type = infer_db_source_type(db, getattr(db, "_source_path", ""))
        if source_type == "csv":
            _CSV_DBS.append(db)
        else:
            _PDF_DBS.append(db)
    if _DBS:
        print(f"Loaded {len(_DBS)} vectorstore(s) from {vs_dir}", file=sys.stderr)
    else:
        print("No existing vectorstores found — creating per-PDF vectorstores.", file=sys.stderr)
        uploaded = []
        for p in pdf_paths:
            try:
                up = core.upload_pdf(p, pdfs_dir)
                uploaded.append(up)
            except Exception as e:
                print(f"Warning: failed to upload {p}: {e}", file=sys.stderr)

        if not uploaded:
            print("No PDFs were successfully uploaded.", file=sys.stderr)
            # If a CSV dataset exists, proceed in CSV-only mode; otherwise abort.
            csv_path_env = os.environ.get("MATERIALS_CSV", os.path.join("/app/storage", "materials", "materials.csv"))
            csv_path_env = os.path.expanduser(csv_path_env)
            if not os.path.exists(csv_path_env):
                print(f"No CSV found at {csv_path_env} and no PDFs uploaded. Aborting.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"CSV found at {csv_path_env}; proceeding in CSV-only mode.", file=sys.stderr)

        for up in uploaded:
            try:
                db, store_dir = core.create_or_load_vector_store(up, vs_dir, embeddings, reindex=reindex)
                _DBS.append(db)
                _PDF_DBS.append(db)
                print(f"Created/loaded vectorstore for {up} -> {store_dir}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to create vectorstore for {up}: {e}", file=sys.stderr)

    # === CSV dataset indexing (add CSV vectorstore to _DBS) ===
    try:
        csv_path_env = os.environ.get("MATERIALS_CSV", None)
        # fallback path used by phase_gen handler
        if not csv_path_env:
            csv_path_env = os.environ.get("MATERIALS_CSV", os.path.join("/app/storage", "materials", "materials_cleaned_shortened_names_as_they_are_FULL.csv"))
            csv_path_env = os.path.expanduser(csv_path_env)

        if os.path.exists(csv_path_env):
            if _CSV_DBS:
                print("CSV vectorstore already loaded from disk; skipping CSV re-indexing.", file=sys.stderr)
                print(
                    f"Initialization finished: {len(_PDF_DBS)} PDF vectorstore(s), {len(_CSV_DBS)} CSV vectorstore(s).",
                    file=sys.stderr,
                )
                return
            print(f"Indexing CSV dataset for RAG: {csv_path_env}", file=sys.stderr)
            # read CSV, create compact text per row
            df_csv = pd.read_csv(csv_path_env)
            texts = []
            metadatas = []
            for idx, row in df_csv.iterrows():
                # compact textualization: "col: val | col2: val2"
                txt = " | ".join([f"{col}: {row[col]}" for col in df_csv.columns])
                texts.append(txt)
                metadatas.append({"source_type": "csv", "row_id": int(idx)})

            if texts:
                # Use the same embeddings instance already created above
                # 'embeddings' variable is created earlier via get_embeddings_provider(...)
                try:
                    from langchain_community.vectorstores import FAISS as _FAISS
                except Exception:
                    _FAISS = None

                if _FAISS is None:
                    print("LangChain FAISS not available; skipping CSV vectorstore creation.", file=sys.stderr)
                else:
                    try:
                        # Build FAISS vectorstore using the existing `embeddings` provider
                        csv_db = _FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                        # persist under vs_dir/csv_index so load_vectorstores_from_dir can find it later
                        try:
                            csv_store_dir = os.path.join(vs_dir, "csv_index")
                            os.makedirs(csv_store_dir, exist_ok=True)
                            csv_db.save_local(csv_store_dir)
                            print(f"Persisted CSV vectorstore to {csv_store_dir}", file=sys.stderr)
                        except Exception as e:
                            print(f"Warning: could not persist CSV vectorstore: {e}", file=sys.stderr)
                        # append to global DB list so combined_retrieve will search it
                        _DBS.append(csv_db)
                        _CSV_DBS.append(csv_db)
                        print("CSV vectorstore added to _DBS.", file=sys.stderr)
                    except Exception as e:
                        print(f"CSV FAISS creation error (continuing): {e}", file=sys.stderr)
        else:
            print(f"CSV file not found at {csv_path_env}; skipping CSV indexing.", file=sys.stderr)
    except Exception as e:
        print(f"CSV indexing error (continuing): {e}", file=sys.stderr)

    if not _DBS:
        # Allow CSV-only operation if a CSV dataset exists (it may be indexed below).
        csv_path_env = os.environ.get("MATERIALS_CSV", os.path.join("/app/storage", "materials", "materials.csv"))
        csv_path_env = os.path.expanduser(csv_path_env)
        if os.path.exists(csv_path_env):
            print(f"No PDF vectorstores, but CSV found at {csv_path_env}. Continuing in CSV-only mode.", file=sys.stderr)
        else:
            print("No vectorstores available after initialization!", file=sys.stderr)
            sys.exit(1)

    # done
    print(
        f"Initialization finished: {len(_PDF_DBS)} PDF vectorstore(s), {len(_CSV_DBS)} CSV vectorstore(s).",
        file=sys.stderr,
    )

@app.on_event("startup")
def startup_event():
    """
    Initialize inside FastAPI worker process. Uses ./pdfs by default and VS_DIR env or default.
    """
    global _DBS, _llm
    if _DBS and _llm is not None:
        return

    pdfs_dir = os.environ.get("PDFS_DIR", PDFS_DIR_DEFAULT)
    vs_dir = os.environ.get("VS_DIR", VS_DIR_DEFAULT)
    sent_model = os.environ.get("SENT_MODEL", DEFAULT_SENT_MODEL)
    # ollama_model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
    reindex = os.environ.get("REINDEX", "false").lower() == "true"

    print("Starting initialization inside FastAPI startup handler...", file=sys.stderr)
    # init_services_from_pdfs(pdfs_dir, vs_dir, sent_model, ollama_model, reindex)
    init_services_from_pdfs(pdfs_dir, vs_dir, sent_model, hf_model, reindex)
    print("Initialization complete (startup handler).", file=sys.stderr)


from fastapi.responses import JSONResponse

from phase_diagram_backend import generate_phase_diagram

# --- Phase diagram generation endpoint (JSON for hvPlot) ---
@app.post("/phase_gen")
async def phase_gen(
    A: str | None = None,
    B: str | None = None,
    C: str | None = None,
    n_steps: int = 101,
    authorization: str | None = Header(None),
):
    # Auth check (same pattern as /generate)
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not (A and B and C):
        raise HTTPException(status_code=400, detail="A, B and C query parameters are required")

    csv_path = os.path.join("/app/storage", "materials", "materials.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=500, detail=f"csv not found at {csv_path}")

    # Run compute in threadpool
    try:
        NDF, CDF = await run_in_threadpool(
            generate_phase_diagram, [A, B, C], csv_path, n_steps, None
        )
    except FileNotFoundError as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    except ValueError as ex:
        raise HTTPException(status_code=404, detail=str(ex))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Internal error during computation: {ex}")

    def _sanitize_df_for_json(df):
        if df is None or df.empty:
            return []
        df2 = df.copy()
        # coerce numeric columns (optional but safe)
        for col in ("x", "T"):
            if col in df2.columns:
                df2[col] = pd.to_numeric(df2[col], errors="coerce")
        # replace infinities with NaN so pandas will convert to null
        df2 = df2.replace([np.inf, -np.inf], np.nan)
        # use pandas -> JSON string (NaN -> null), then parse to Python objects
        txt = df2.to_json(orient="records", date_format="iso", double_precision=10)
        return json.loads(txt)
    
    return JSONResponse(
        content={
            "neel": _sanitize_df_for_json(NDF),
            "curie": _sanitize_df_for_json(CDF),
        },
        headers={"X-Log-Mode": "server_compute"},
    )

# ----------------- CLI convenience -----------------

def parse_args():
    p = argparse.ArgumentParser(description="PDF Chat model server (folder-driven)")
    p.add_argument("--pdfs-dir", default=PDFS_DIR_DEFAULT, help="Directory containing PDFs (default ./pdfs)")
    p.add_argument("--vs-dir", default=VS_DIR_DEFAULT, help="Vectorstore directory")
    # p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    p.add_argument("--sent-model", default=DEFAULT_SENT_MODEL)
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # init_services_from_pdfs(args.pdfs_dir, args.vs_dir, args.sent_model, args.ollama_model, args.reindex)
    init_services_from_pdfs(args.pdfs_dir, args.vs_dir, args.sent_model, args.hf_model, args.reindex)
    print("Initialization complete. Starting server on http://%s:%d" % (args.host, args.port))
    uvicorn.run("pdf_chatbot_server:app", host=args.host, port=args.port, log_level="info")
