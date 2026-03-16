#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# embeddings.py
import os
from typing import List, Any

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "sentence-transformers is required for embeddings.\n"
        "Install with: pip install sentence-transformers"
    ) from e


# ---- HARD-CODED CACHE DIRECTORY ----
HF_CACHE_DIR = os.path.join("/app/storage", "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)


class SentenceTransformersEmbeddings:
    """
    Sentence-transformers embeddings wrapper compatible with
    LangChain FAISS and other vector stores.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Model will be downloaded ONCE and cached here
        self.model = SentenceTransformer(
            model_name,
            cache_folder=HF_CACHE_DIR
        )

    def _normalize_inputs(self, items: Any) -> List[str]:
        """
        Normalize input to list[str].
        Supports:
          - str
          - list[str]
          - list[Document-like objects with .page_content or .text]
        """
        if isinstance(items, str):
            return [items]

        if isinstance(items, list):
            texts = []
            for it in items:
                if isinstance(it, str):
                    texts.append(it)
                else:
                    text = getattr(it, "page_content", None)
                    if text is None:
                        text = getattr(it, "text", None)
                    if text is None:
                        text = str(it)
                    texts.append(text)
            return texts

        return [str(items)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    def __call__(self, items: Any):
        """
        Make the object callable to support older / variant LangChain codepaths.
        """
        if isinstance(items, str):
            return self.embed_query(items)

        if isinstance(items, list):
            texts = self._normalize_inputs(items)
            return self.embed_documents(texts)

        return self.embed_query(str(items))


def get_embeddings_provider(model_name: str = "all-MiniLM-L6-v2"):
    """
    Factory kept for API stability with main.py.
    """
    return SentenceTransformersEmbeddings(model_name=model_name)
