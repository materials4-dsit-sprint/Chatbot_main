#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding utilities and wrappers used by the chatbot and server pipelines.
"""

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
        """
        Initialize the sentence-transformer model wrapper.

        Parameters
        ----------
        model_name : str, optional
            Name of the sentence-transformer model to load, by default "all-MiniLM-L6-v2".

        Returns
        -------
        None
            This initializer stores the loaded model on the instance.
        """
        # Model will be downloaded ONCE and cached here
        self.model = SentenceTransformer(
            model_name,
            cache_folder=HF_CACHE_DIR
        )

    def _normalize_inputs(self, items: Any) -> List[str]:
        """
        Normalize supported input types into a list of strings.

        Parameters
        ----------
        items : Any
            Single string, list of strings, or document-like objects.

        Returns
        -------
        List[str]
            Normalized text values ready for embedding.
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
        """
        Embed multiple text documents.

        Parameters
        ----------
        texts : List[str]
            Input text documents to embed.

        Returns
        -------
        List[List[float]]
            Embedding vectors for each input text.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        Parameters
        ----------
        text : str
            Query text to embed.

        Returns
        -------
        List[float]
            Embedding vector for the query.
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

    def __call__(self, items: Any):
        """
        Provide a callable interface for embedding queries or document lists.

        Parameters
        ----------
        items : Any
            Single text, list of texts, or document-like inputs.

        Returns
        -------
        Any
            Query embedding or list of document embeddings depending on the input.
        """
        if isinstance(items, str):
            return self.embed_query(items)

        if isinstance(items, list):
            texts = self._normalize_inputs(items)
            return self.embed_documents(texts)

        return self.embed_query(str(items))


def get_embeddings_provider(model_name: str = "all-MiniLM-L6-v2"):
    """
    Create the configured embeddings provider.

    Parameters
    ----------
    model_name : str, optional
        Name of the sentence-transformer model to load, by default "all-MiniLM-L6-v2".

    Returns
    -------
    SentenceTransformersEmbeddings
        Initialized embeddings wrapper instance.
    """
    return SentenceTransformersEmbeddings(model_name=model_name)
