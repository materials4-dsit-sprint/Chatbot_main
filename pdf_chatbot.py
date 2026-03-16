#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_chatbot

PDF-based chatbot using sentence-transformers for embeddings,
FAISS for retrieval, and Ollama for generation.
"""

# pdf_chatbot.py
import sys
import os
import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from embeddings import get_embeddings_provider
import core

PDFS_DIR_DEFAULT = os.path.join("/app/storage", "pdfs")
VS_DIR_DEFAULT = os.path.join("/app/storage", "pdf_vectorstores")
DEFAULT_OLLAMA_MODEL = "deepseek-r1:8b"
DEFAULT_SENT_MODEL = "all-MiniLM-L6-v2"

"""You are an assistant that answers questions using the retrieved context.
If you do not know the answer, say you do not know.
Use at most ten sentences."""

TEMPLATE = """
You are an expert assistant who follows user instructions in the question, thinks and then answers questions only using the retrieved context. Do not invent facts or guess beyond the provided context — if the answer is not present, say “I don’t know.” Answer concisely (max thirty sentences). After the answer, list the sources you used in the format [source_type: identifier] (e.g., [csv: row 123] or [pdf: filename page X]) so the user can verify.

Question: {question}
Context: {context}
Answer:
"""

# def build_prompt(question: str, docs) -> str:
#     context = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
#     prompt = ChatPromptTemplate.from_template(TEMPLATE)
#     # We will use the prompt template only to format the string for the LLM invocation.
#     return TEMPLATE.format(question=question, context=context)

def build_prompt(question: str, docs) -> str:
    context_blocks = []

    for d in docs:
        content = getattr(d, "page_content", str(d))
        metadata = getattr(d, "metadata", {}) or {}

        # Determine source label
        source_type = metadata.get("source_type")
        if source_type == "csv":
            label = f"[csv: row {metadata.get('row_id', 'unknown')}]"
        else:
            # default to pdf-style labeling
            source = metadata.get("source", "pdf")
            page = metadata.get("page")
            if page is not None:
                label = f"[pdf: {os.path.basename(source)} page {page}]"
            else:
                label = f"[pdf: {os.path.basename(source)}]"

        context_blocks.append(f"{label}\n{content}")

    context = "\n\n".join(context_blocks)

    return TEMPLATE.format(question=question, context=context)

def run_repl(db, llm, k):
    print("\nPDF loaded. Ask questions (type 'bye' to exit).")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if q.lower() in ("bye", "exit", "quit"):
            print("Goodbye.")
            break

        docs = core.retrieve_docs(db, q, k=k)
        if not docs:
            print("No relevant documents found.")
            continue

        prompt = build_prompt(q, docs)
        # Use the robust invoker that handles many LangChain/LLM return shapes
        try:
            text = invoke_llm_and_get_text(llm, prompt)
            print("\nAnswer:\n", text.strip())
        except Exception as e:
            print("LLM call failed:", e)


def parse_args():
    p = argparse.ArgumentParser(description="PDF Chat (minimal main)")
    p.add_argument("pdf", help="Path to PDF")
    p.add_argument("--pdfs-dir", default=PDFS_DIR_DEFAULT)
    p.add_argument("--vs-dir", default=VS_DIR_DEFAULT)
    p.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    p.add_argument("--sent-model", default=DEFAULT_SENT_MODEL)
    p.add_argument("--reindex", action="store_true")
    p.add_argument("-k", type=int, default=30, help="Number of retrieved chunks")
    return p.parse_args()

def invoke_llm_and_get_text(llm, prompt_text):
    """
    Try common LLM call patterns and return a single string response.
    Works for LLMs that implement __call__, .generate, .predict, or .invoke-like APIs.
    """
    # 1) If llm is callable (has __call__), try it and handle different return shapes
    try:
        if callable(llm):
            res = llm(prompt_text)
            # If a raw string was returned
            if isinstance(res, str):
                return res
            # LangChain LLMResult-like object with .generations
            if hasattr(res, "generations"):
                gens = res.generations
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    # first candidate
                    candidate = gens[0][0]
                    # text might be .text or .content
                    return getattr(candidate, "text", getattr(candidate, "content", str(candidate)))
            # If it's a dict or list, try sensible extraction
            if isinstance(res, dict) and "text" in res:
                return res["text"]
            # fallback to string form
            return str(res)
    except TypeError:
        # not callable
        pass
    except Exception as e:
        # callable but failed — try other invocations below, but keep message
        last_exc = e

    # 2) Try .generate([prompt])
    try:
        if hasattr(llm, "generate"):
            gen = llm.generate([prompt_text])
            if hasattr(gen, "generations"):
                gens = gen.generations
                if gens and len(gens) > 0 and len(gens[0]) > 0:
                    return getattr(gens[0][0], "text", str(gens[0][0]))
            return str(gen)
    except Exception as e:
        last_exc = e

    # 3) Try .predict or .complete or .invoke methods
    for method_name in ("predict", "complete", "invoke", "call"):
        try:
            method = getattr(llm, method_name, None)
            if callable(method):
                out = method(prompt_text)
                if isinstance(out, str):
                    return out
                # try similar extraction as above
                if hasattr(out, "generations"):
                    gens = out.generations
                    if gens and len(gens) > 0 and len(gens[0]) > 0:
                        return getattr(gens[0][0], "text", str(gens[0][0]))
                if isinstance(out, dict) and "text" in out:
                    return out["text"]
                return str(out)
        except Exception as e:
            last_exc = e

    # 4) Give a helpful error if none of the above worked
    raise RuntimeError(
        "Unable to invoke LLM with known call patterns. "
        "Tried __call__, .generate, .predict, .complete, .invoke. "
        f"Last error: {repr(last_exc)}"
    )


def main():
    args = parse_args()

    # ---- Validate PDF path ----
    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        print(f"Invalid PDF path: {pdf_path}")
        sys.exit(1)

    # ---- Initialize embeddings (sentence-transformers only) ----
    try:
        embeddings = get_embeddings_provider(model_name=args.sent_model)
    except Exception as e:
        print("Failed to initialize sentence-transformers embeddings:")
        print(e)
        sys.exit(1)

    # ---- Initialize Ollama LLM (generation only) ----
    try:
        llm = OllamaLLM(model=args.ollama_model)
    except Exception as e:
        print("Failed to initialize Ollama LLM:")
        print("Ensure Ollama is running and the model exists (check `ollama list`).")
        print(e)
        sys.exit(1)

    # ---- Upload / copy PDF into managed directory ----
    try:
        pdf_dest = core.upload_pdf(pdf_path, args.pdfs_dir)
    except Exception as e:
        print("Failed to upload PDF:")
        print(e)
        sys.exit(1)

    # ---- Create or load vector store ----
    try:
        db, store_dir = core.create_or_load_vector_store(
            pdf_dest,
            args.vs_dir,
            embeddings,
            reindex=args.reindex
        )
    except Exception as e:
        print("Failed to create or load vector store:")
        print(e)
        sys.exit(1)

    # ---- Interactive loop ----
    try:
        run_repl(db, llm, k=args.k)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception as e:
        print("Unexpected runtime error:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
