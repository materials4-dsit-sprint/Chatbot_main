#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for constructing Hugging Face text-generation pipelines.
"""

import sys
from typing import Any

import torch
from transformers import BitsAndBytesConfig, pipeline


def build_text_generation_pipeline(model_name: str, **pipeline_kwargs: Any):
    """
    Build a text-generation pipeline for the requested model.

    Parameters
    ----------
    model_name : str
        Name of the Hugging Face model to load.
    **pipeline_kwargs : Any
        Additional keyword arguments forwarded to `transformers.pipeline`.

    Returns
    -------
    Any
        Configured text-generation pipeline instance.
    """
    if not torch.cuda.is_available():
        print(
            f"[hf] CUDA unavailable; loading {model_name} without 4-bit quantization.",
            file=sys.stderr,
        )
        return pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            **pipeline_kwargs,
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    return pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        model_kwargs={
            "quantization_config": quantization_config,
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
        },
        **pipeline_kwargs,
    )
