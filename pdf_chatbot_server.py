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
from pathlib import Path
from math import ceil
from typing import Literal
from fastapi import FastAPI, HTTPException, Header, File, UploadFile, Form
from pydantic import BaseModel
import uvicorn
from starlette.concurrency import run_in_threadpool
from pdf_chatbot import build_prompt, invoke_llm_and_get_text
from embeddings import get_embeddings_provider
import core
import numpy as np
import pandas as pd
import json
from fastapi.responses import JSONResponse, StreamingResponse
from llm_runtime import build_llm, get_active_pipeline, get_configured_default_model, get_ollama_base_url, resolve_model_selection

# Defaults
PDFS_DIR_DEFAULT = os.path.join("/app/storage", "pdfs")
VS_DIR_DEFAULT = os.path.join("/app/storage", "pdf_vectorstores")
MATERIALS_DIR_DEFAULT = os.path.join("/app/storage", "materials")
CSV_VS_DIR_DEFAULT = os.path.join("/app/storage", "csv_vectorstores")
DEFAULT_SENT_MODEL = "all-MiniLM-L6-v2"
app = FastAPI()

# import the router
from llm_phase_diagram_gen import router as llm_phase_gen_router
app.include_router(llm_phase_gen_router, prefix="")


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
_embeddings = None
# --- LLM cache for model switching ---
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
        prepared = prepare_generation(req)
        if prepared["terminal_text"] is not None:
            return {"text": prepared["terminal_text"]}

        text = generate_answer_text(prepared, req)
        return {"text": text}
    result = await run_in_threadpool(work)
    return result


@app.post("/generate-stream")
async def generate_stream(req: GenReq, authorization: str | None = Header(None)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not _DBS or _llm is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    def event_stream():
        try:
            prepared = prepare_generation(req)

            if prepared["retrieved"]:
                yield json.dumps({
                    "event": "retrieval",
                    "context_source": req.context_source,
                    "k_requested": prepared["k_used"],
                    "retrieved_count": len(prepared["retrieved"]),
                    "retrieved": prepared["retrieved"],
                }, ensure_ascii=False) + "\n"

            if prepared["terminal_text"] is not None:
                yield json.dumps({
                    "event": "answer",
                    "text": prepared["terminal_text"],
                }, ensure_ascii=False) + "\n"
                return

            text = generate_answer_text(prepared, req)
            yield json.dumps({
                "event": "answer",
                "text": text,
                "model_used": prepared["selected_model"],
            }, ensure_ascii=False) + "\n"
        except HTTPException as e:
            yield json.dumps({
                "event": "error",
                "status_code": e.status_code,
                "detail": e.detail,
            }, ensure_ascii=False) + "\n"
        except Exception as e:
            print(f"Streaming generation failed: {e}", file=sys.stderr)
            yield json.dumps({
                "event": "error",
                "status_code": 500,
                "detail": str(e),
            }, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _normalize_chunk_text(text: str, limit: int = 1200) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _build_retrieved_summary(docs) -> list[dict]:
    retrieved_summary = []
    for idx, d in enumerate(docs, start=1):
        md = getattr(d, "metadata", {}) or {}
        source = md.get("source")
        filename = md.get("filename")
        if not filename and source:
            filename = os.path.basename(source)

        row_id = md.get("row_id")
        retrieved_summary.append({
            "rank": idx,
            "source_type": md.get("source_type", "pdf"),
            "id": row_id if row_id is not None else (source or filename or idx),
            "source": source,
            "filename": filename,
            "page": md.get("page"),
            "snippet": _normalize_chunk_text(getattr(d, "page_content", "")),
        })
    return retrieved_summary


def _log_retrieval(req: GenReq, request_id: str, k_used: int, retrieved_summary: list[dict]) -> None:
    append_chat_log({
        "event": "retrieval",
        "request_id": request_id,
        "question": req.question,
        "context_source": req.context_source,
        "k_requested": k_used,
        "retrieved_count": len(retrieved_summary),
        "retrieved": retrieved_summary,
    })


def _get_llm_instance(selected_model: str):
    model_details = resolve_model_selection(selected_model, strict=False)

    actual_model_name = str(model_details["actual_model_name"])
    if actual_model_name not in _LLM_CACHE:
        print(
            f"Initializing new LLM instance: pipeline={get_active_pipeline()} model={actual_model_name}",
            file=sys.stderr,
        )
        _resolved, llm_instance = build_llm(selected_model, max_new_tokens=1024)
        _LLM_CACHE[actual_model_name] = llm_instance

    return model_details, _LLM_CACHE[actual_model_name]


def prepare_generation(req: GenReq) -> dict:
    k_used = req.k or 30
    selected_dbs = get_dbs_for_context_source(req.context_source)
    if not selected_dbs:
        source_label = "PDF" if req.context_source == "pdfs" else "CSV"
        return {
            "terminal_text": f"No indexed {source_label} sources are available.",
            "retrieved": [],
            "k_used": k_used,
        }

    docs = combined_retrieve(selected_dbs, req.question, k_used)
    if not docs:
        return {
            "terminal_text": "No relevant documents found.",
            "retrieved": [],
            "k_used": k_used,
        }

    rid = str(uuid.uuid4())
    retrieved_summary = _build_retrieved_summary(docs)

    if req.log:
        _log_retrieval(req, rid, k_used, retrieved_summary)

    default_model = get_configured_default_model()
    selected_model = req.model or str(default_model["model_key"] or default_model["actual_model_name"])
    model_details, llm_instance = _get_llm_instance(selected_model)

    return {
        "terminal_text": None,
        "request_id": rid,
        "k_used": k_used,
        "docs": docs,
        "retrieved": retrieved_summary,
        "prompt": build_prompt(req.question, docs),
        "selected_model": str(model_details["actual_model_name"]),
        "selected_model_key": str(model_details["model_key"] or selected_model),
        "llm_instance": llm_instance,
    }


def generate_answer_text(prepared: dict, req: GenReq) -> str:
    if prepared["terminal_text"] is not None:
        return prepared["terminal_text"]

    text = invoke_llm_and_get_text(prepared["llm_instance"], prepared["prompt"])

    if req.log:
        append_chat_log({
            "event": "answer",
            "request_id": prepared["request_id"],
            "question": req.question,
            "context_source": req.context_source,
            "k_used": prepared["k_used"],
            "llm_answer": text,
            "context_length_chars": len(prepared["prompt"]),
            "model_used": prepared["selected_model"],
            "model_key": prepared["selected_model_key"],
        })

    return text

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

def _replace_registered_db(target_list: list, new_db, source_path: str):
    source_path = os.path.abspath(source_path)
    kept = []
    for existing_db in target_list:
        existing_path = os.path.abspath(getattr(existing_db, "_source_path", ""))
        if existing_path != source_path:
            kept.append(existing_db)
    kept.append(new_db)
    target_list[:] = kept

def register_vectorstore(db, source_type: str, source_path: str):
    global _DBS, _PDF_DBS, _CSV_DBS

    source_path = os.path.abspath(source_path)
    setattr(db, "_source_path", source_path)
    normalized_source_type = "csv" if source_type == "csv" else "pdf"

    _replace_registered_db(_DBS, db, source_path)
    if normalized_source_type == "csv":
        _replace_registered_db(_CSV_DBS, db, source_path)
    else:
        _replace_registered_db(_PDF_DBS, db, source_path)

def _save_uploaded_file(upload: UploadFile, dest_dir: str, allowed_suffixes: set[str]) -> str:
    filename = os.path.basename(upload.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    return dest_path

def build_csv_vectorstore(csv_path: str, vs_root: str, embeddings, reindex: bool = False):
    from langchain_community.vectorstores import FAISS as _FAISS

    os.makedirs(vs_root, exist_ok=True)
    base_name = os.path.basename(csv_path)
    store_dir = os.path.join(vs_root, base_name + "_faiss")

    if reindex and os.path.exists(store_dir):
        import shutil
        shutil.rmtree(store_dir)

    if os.path.exists(store_dir):
        db = _FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)
        return db, store_dir

    df_csv = pd.read_csv(csv_path)
    texts = []
    metadatas = []
    for idx, row in df_csv.iterrows():
        txt = " | ".join([f"{col}: {row[col]}" for col in df_csv.columns])
        texts.append(txt)
        metadatas.append({
            "source_type": "csv",
            "row_id": int(idx),
            "source": csv_path,
            "filename": base_name,
        })

    if not texts:
        raise RuntimeError(f"No rows found in CSV: {csv_path}")

    db = _FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    db.save_local(store_dir)
    return db, store_dir

def combined_retrieve(dbs, query: str, k_total: int):
    """
    Query multiple DBs and merge results with global ranking across stores.
    Returns up to k_total documents.
    """
    if not dbs:
        return []

    # request more per DB to allow for dedup/uneven coverage
    overfetch_factor = 2   # try 2x; increase if needed
    k_per_db = max(1, ceil((k_total * overfetch_factor) / len(dbs)))
    collected = []

    for db in dbs:
        docs_with_scores = []
        try:
            if hasattr(core, "retrieve_docs_with_scores"):
                docs_with_scores = core.retrieve_docs_with_scores(db, query, k=k_per_db) or []
            elif hasattr(db, "similarity_search_with_score"):
                docs_with_scores = db.similarity_search_with_score(query, k=k_per_db) or []
            elif hasattr(db, "similarity_search"):
                docs = db.similarity_search(query, k=k_per_db) or []
                docs_with_scores = [(doc, float(rank)) for rank, doc in enumerate(docs, start=1)]
            else:
                docs_with_scores = []
        except Exception as e:
            print(f"Warning: retrieval from a DB failed: {e}", file=sys.stderr)
            docs_with_scores = []

        for d, score in docs_with_scores:
            text = getattr(d, "page_content", None)
            if text is None:
                try:
                    text = str(d)
                except Exception:
                    text = ""
            metadata = getattr(d, "metadata", {}) or {}
            key = (
                metadata.get("source"),
                metadata.get("page"),
                metadata.get("start_index"),
                text.strip()[:1024],
            )
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = float("inf")
            collected.append((numeric_score, key, d))

    best_docs = {}
    for score, key, doc in collected:
        if not key[-1]:
            continue
        existing = best_docs.get(key)
        if existing is None or score < existing[0]:
            best_docs[key] = (score, doc)

    ranked = sorted(best_docs.values(), key=lambda item: item[0])
    deduped = [doc for _score, doc in ranked[:k_total]]
    print(
        f"[retrieve] requested k_total={k_total}, dbs={len(dbs)}, collected={len(collected)}, deduped={len(deduped)}",
        file=sys.stderr,
    )
    return deduped

@app.post("/upload-context")
async def upload_context(
    context_source: Literal["pdfs", "csvs"] = Form(...),
    file: UploadFile = File(...),
    authorization: str | None = Header(None),
):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if _llm is None or _embeddings is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    if context_source == "pdfs":
        dest_dir = os.environ.get("PDFS_DIR", PDFS_DIR_DEFAULT)
        allowed_suffixes = {".pdf"}
        vs_root = os.environ.get("VS_DIR", VS_DIR_DEFAULT)
    else:
        dest_dir = os.environ.get("MATERIALS_DIR", MATERIALS_DIR_DEFAULT)
        allowed_suffixes = {".csv"}
        vs_root = os.environ.get("CSV_VS_DIR", CSV_VS_DIR_DEFAULT)

    dest_path = _save_uploaded_file(file, dest_dir, allowed_suffixes)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    def index_uploaded_file():
        if context_source == "pdfs":
            db, store_dir = core.create_or_load_vector_store(dest_path, vs_root, _embeddings, reindex=True)
            register_vectorstore(db, "pdf", store_dir)
        else:
            db, store_dir = build_csv_vectorstore(dest_path, vs_root, _embeddings, reindex=True)
            register_vectorstore(db, "csv", store_dir)

        return {
            "ok": True,
            "filename": os.path.basename(dest_path),
            "stored_at": dest_path,
            "vectorstore_path": store_dir,
            "context_source": context_source,
            "pdf_vectorstores": len(_PDF_DBS),
            "csv_vectorstores": len(_CSV_DBS),
        }

    try:
        return await run_in_threadpool(index_uploaded_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index uploaded file: {e}")

# ----------------- initialization -----------------

def init_services_from_pdfs(
    pdfs_dir: str,
    vs_dir: str,
    sent_model: str,
    selected_model: str | None,
    reindex: bool,
):
    """
    Initialize embeddings, load/create vectorstores (one per saved VS or per PDF),
    and initialize the configured LLM. Populates global _DBS and _llm.
    """
    global _DBS, _PDF_DBS, _CSV_DBS, _llm, _embeddings

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
        _embeddings = embeddings
    except Exception as e:
        print("Failed to initialize sentence-transformers embeddings:", e, file=sys.stderr)
        sys.exit(1)

    try:
        if get_active_pipeline() == "ollama":
            print(
                f"Using Ollama endpoint: {get_ollama_base_url() or 'http://127.0.0.1:11434'}",
                file=sys.stderr,
            )
        model_details, _llm = build_llm(selected_model, max_new_tokens=1024)
        _LLM_CACHE[str(model_details["actual_model_name"])] = _llm
    except Exception as e:
        print(f"Failed to initialize {get_active_pipeline()} LLM: {e}", file=sys.stderr)
        if get_active_pipeline() == "ollama":
            print("Make sure Ollama is running and the model exists.", file=sys.stderr)
        sys.exit(1)

    # Try loading existing vectorstores from disk
    _DBS = []
    _PDF_DBS = []
    _CSV_DBS = []

    for db in load_vectorstores_from_dir(vs_dir, embeddings):
        register_vectorstore(db, infer_db_source_type(db, getattr(db, "_source_path", "")), getattr(db, "_source_path", vs_dir))

    csv_vs_dir = os.environ.get("CSV_VS_DIR", CSV_VS_DIR_DEFAULT)
    for db in load_vectorstores_from_dir(csv_vs_dir, embeddings):
        register_vectorstore(db, "csv", getattr(db, "_source_path", csv_vs_dir))

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
                register_vectorstore(db, "pdf", store_dir)
                print(f"Created/loaded vectorstore for {up} -> {store_dir}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to create vectorstore for {up}: {e}", file=sys.stderr)

    # === CSV dataset indexing (add CSV vectorstore to _DBS) ===
    try:
        csv_path_env = os.environ.get("MATERIALS_CSV", None)
        # fallback path used by the script phase generator handler
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
                        csv_vs_dir = os.environ.get("CSV_VS_DIR", CSV_VS_DIR_DEFAULT)
                        os.makedirs(csv_vs_dir, exist_ok=True)

                        # Keep the default startup CSV indexed exactly as before, but
                        # persist it with the CSV vector stores instead of mixing it
                        # into the PDF vector store directory.
                        csv_db = _FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                        csv_store_dir = os.path.join(
                            csv_vs_dir,
                            os.path.basename(csv_path_env) + "_faiss",
                        )
                        try:
                            csv_db.save_local(csv_store_dir)
                            print(f"Persisted CSV vectorstore to {csv_store_dir}", file=sys.stderr)
                        except Exception as e:
                            print(f"Warning: could not persist CSV vectorstore: {e}", file=sys.stderr)

                        register_vectorstore(csv_db, "csv", csv_store_dir)
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
    default_model = get_configured_default_model()
    reindex = os.environ.get("REINDEX", "false").lower() == "true"

    print(
        f"Starting initialization inside FastAPI startup handler with WHICH_PIPELINE={get_active_pipeline()}...",
        file=sys.stderr,
    )
    init_services_from_pdfs(
        pdfs_dir,
        vs_dir,
        sent_model,
        str(default_model["model_key"] or default_model["actual_model_name"]),
        reindex,
    )
    print("Initialization complete (startup handler).", file=sys.stderr)


from script_phase_diagram_gen import generate_phase_diagram

# --- Phase diagram generation endpoint (JSON for hvPlot) ---
@app.post("/script_phase_gen")
async def script_phase_gen_endpoint(
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
    p.add_argument("--hf-model", default=os.environ.get("HF_MODEL"))
    p.add_argument("--ollama-model", default=os.environ.get("OLLAMA_MODEL"))
    p.add_argument("--sent-model", default=DEFAULT_SENT_MODEL)
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cli_model = args.ollama_model if get_active_pipeline() == "ollama" else args.hf_model
    init_services_from_pdfs(args.pdfs_dir, args.vs_dir, args.sent_model, cli_model, args.reindex)
    print("Initialization complete. Starting server on http://%s:%d" % (args.host, args.port))
    uvicorn.run("pdf_chatbot_server:app", host=args.host, port=args.port, log_level="info")
