#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cb_core.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

def upload_pdf(src_path: str, dest_dir: str) -> str:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"PDF not found: {src_path}")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src_path))
    if os.path.abspath(src_path) != os.path.abspath(dest):
        with open(src_path, "rb") as fsrc, open(dest, "wb") as fdst:
            fdst.write(fsrc.read())
    return dest

def create_or_load_vector_store(pdf_path: str, vs_root: str, embeddings, reindex: bool = False):
    """
    Create or load a FAISS vector store for a single PDF.
    embeddings must implement embed_documents/embed_query (see cb_embeddings.py)
    """
    os.makedirs(vs_root, exist_ok=True)
    name = os.path.basename(pdf_path)
    store_dir = os.path.join(vs_root, name + "_faiss")

    if reindex and os.path.exists(store_dir):
        import shutil
        shutil.rmtree(store_dir)

    if os.path.exists(store_dir):
        # load existing vector store (only if you trust the files)
        db = FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)

        return db, store_dir

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise RuntimeError("No pages loaded from PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    try:
        db.save_local(store_dir)
    except Exception:
        # not fatal; continue using in-memory index
        pass
    return db, store_dir

# # def retrieve_docs(db, query: str, k: int = 4) -> List[Document]:
# def retrieve_docs(db, query: str, k: int = 4):
#     # handle different langchain signatures gracefully
#     try:
#         return db.similarity_search(query, k=k)
#     except TypeError:
#         return db.similarity_search(query, k)

def retrieve_docs(db, query: str, k: int = 4):
    return [doc for doc, _score in retrieve_docs_with_scores(db, query, k=k)]


def retrieve_docs_with_scores(db, query: str, k: int = 4):
    """
    Hybrid retrieval:
    1) Extract chemical-like formulas via regex
    2) Exact substring match (regex hits)
    3) Semantic FAISS similarity search
    4) Merge (regex results first, deduplicated)
    """

    # -------- 1. Extract chemical-like formulas --------
    # Matches patterns like La0.7Sr0.3MnO3, La2/3Sr1/3MnO3 etc.
    formula_pattern = r"\b(?:[A-Z][a-z]?\d*(?:\.\d+|/\d+)*)+\b"
    formulas = re.findall(formula_pattern, query)

    # -------- 2. Get ALL documents from FAISS --------
    # Access internal docstore safely
    all_docs = []
    if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
        all_docs = list(db.docstore._dict.values())

    # Exact substring matches
    regex_hits = []
    for doc in all_docs:
        content = getattr(doc, "page_content", "")
        if any(f in content for f in formulas):
            regex_hits.append(doc)

    # -------- 3. Semantic search --------
    semantic_hits_with_scores = []
    try:
        semantic_hits_with_scores = db.similarity_search_with_score(query, k=k)
    except TypeError:
        semantic_hits_with_scores = db.similarity_search_with_score(query, k)
    except AttributeError:
        try:
            semantic_hits = db.similarity_search(query, k=k)
        except TypeError:
            semantic_hits = db.similarity_search(query, k)
        semantic_hits_with_scores = [(doc, float(rank)) for rank, doc in enumerate(semantic_hits, start=1)]

    # -------- 4. Merge (regex first, deduplicated) --------
    ranked_hits = []
    for doc in regex_hits:
        ranked_hits.append((doc, float("-inf")))

    for doc, score in semantic_hits_with_scores:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = float("inf")
        ranked_hits.append((doc, numeric_score))

    best_by_key = {}
    for doc, score in ranked_hits:
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("start_index"),
            getattr(doc, "page_content", "")[:512],
        )
        existing = best_by_key.get(key)
        if existing is None or score < existing[1]:
            best_by_key[key] = (doc, score)

    ordered = sorted(best_by_key.values(), key=lambda item: item[1])
    return ordered[:k]
