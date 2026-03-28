from __future__ import annotations

import os
from typing import Any

from hf_utils import build_text_generation_pipeline

PIPELINE_HF = "hf"
PIPELINE_OLLAMA = "ollama"
SUPPORTED_PIPELINES = {PIPELINE_HF, PIPELINE_OLLAMA}
DEFAULT_MODEL_KEY = "deepseek_1_5b"

MODEL_SPECS: dict[str, dict[str, str]] = {
    "deepseek_1_5b": {
        "label": "DeepSeek:1.5B",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "ollama": "deepseek-r1:1.5b",
    },
    "deepseek_7b": {
        "label": "DeepSeek:7B",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "ollama": "deepseek-r1:7b",
    },
    "deepseek_8b": {
        "label": "DeepSeek:8B",
        "hf": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "ollama": "deepseek-r1:8b",
    },
    "qwen_1_5b": {
        "label": "Qwen:1.5B",
        "hf": "Qwen/Qwen2.5-1.5B-Instruct",
        "ollama": "qwen2.5:1.5b",
    },
    "qwen_3b": {
        "label": "Qwen:3B",
        "hf": "Qwen/Qwen2.5-3B-Instruct",
        "ollama": "qwen2.5:3b",
    },
}


def get_active_pipeline() -> str:
    pipeline = os.environ.get("WHICH_PIPELINE", PIPELINE_HF).strip().lower()
    if pipeline not in SUPPORTED_PIPELINES:
        print(f"[llm_runtime] Unsupported WHICH_PIPELINE={pipeline!r}; falling back to '{PIPELINE_HF}'.")
        return PIPELINE_HF
    return pipeline


def get_model_options() -> dict[str, str]:
    return {spec["label"]: model_key for model_key, spec in MODEL_SPECS.items()}


def get_model_label(model_key: str) -> str:
    return MODEL_SPECS[model_key]["label"]


def get_pipeline_model_name(model_key: str, pipeline: str | None = None) -> str:
    selected_pipeline = pipeline or get_active_pipeline()
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_SPECS[model_key][selected_pipeline]


def get_default_model_key(configured_model: str | None = None, pipeline: str | None = None) -> str:
    if configured_model:
        resolved = resolve_model_selection(configured_model, pipeline=pipeline, strict=False)
    else:
        resolved = get_configured_default_model(pipeline=pipeline)

    if resolved["model_key"]:
        return str(resolved["model_key"])
    return DEFAULT_MODEL_KEY


def get_configured_default_model(
    *,
    hf_env_vars: tuple[str, ...] = ("HF_MODEL",),
    ollama_env_vars: tuple[str, ...] = ("OLLAMA_MODEL",),
    fallback_key: str = DEFAULT_MODEL_KEY,
    pipeline: str | None = None,
) -> dict[str, str | None]:
    selected_pipeline = pipeline or get_active_pipeline()
    env_vars = hf_env_vars if selected_pipeline == PIPELINE_HF else ollama_env_vars

    for env_var in env_vars:
        env_value = os.environ.get(env_var)
        if env_value:
            return resolve_model_selection(env_value, pipeline=selected_pipeline, strict=False)

    return resolve_model_selection(fallback_key, pipeline=selected_pipeline, strict=False)


def resolve_model_selection(
    selection: str | None,
    *,
    pipeline: str | None = None,
    strict: bool = True,
) -> dict[str, str | None]:
    selected_pipeline = pipeline or get_active_pipeline()
    chosen = (selection or "").strip()

    if not chosen:
        model_key = DEFAULT_MODEL_KEY
        return {
            "model_key": model_key,
            "label": get_model_label(model_key),
            "actual_model_name": get_pipeline_model_name(model_key, selected_pipeline),
        }

    if chosen in MODEL_SPECS:
        return {
            "model_key": chosen,
            "label": get_model_label(chosen),
            "actual_model_name": get_pipeline_model_name(chosen, selected_pipeline),
        }

    for model_key, spec in MODEL_SPECS.items():
        if chosen in (spec["hf"], spec["ollama"]):
            return {
                "model_key": model_key,
                "label": spec["label"],
                "actual_model_name": spec[selected_pipeline] if chosen != spec[selected_pipeline] else chosen,
            }

    if strict:
        raise ValueError(f"Unknown model selection: {selection}")

    return {
        "model_key": None,
        "label": chosen,
        "actual_model_name": chosen,
    }


def _import_ollama_llm():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM
    except ImportError:
        from langchain_ollama.llms import OllamaLLM
        return OllamaLLM


def get_ollama_base_url() -> str | None:
    raw_value = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    if not raw_value:
        return None

    value = raw_value.strip()
    if not value:
        return None

    if "://" not in value:
        value = f"http://{value}"
    return value


def build_llm(
    selection: str | None = None,
    *,
    pipeline: str | None = None,
    hf_env_vars: tuple[str, ...] = ("HF_MODEL",),
    ollama_env_vars: tuple[str, ...] = ("OLLAMA_MODEL",),
    fallback_key: str = DEFAULT_MODEL_KEY,
    **kwargs: Any,
):
    selected_pipeline = pipeline or get_active_pipeline()
    if selection:
        model_details = resolve_model_selection(selection, pipeline=selected_pipeline, strict=False)
    else:
        model_details = get_configured_default_model(
            hf_env_vars=hf_env_vars,
            ollama_env_vars=ollama_env_vars,
            fallback_key=fallback_key,
            pipeline=selected_pipeline,
        )

    actual_model_name = str(model_details["actual_model_name"])

    if selected_pipeline == PIPELINE_OLLAMA:
        OllamaLLM = _import_ollama_llm()
        ollama_kwargs: dict[str, Any] = {}
        if "temperature" in kwargs:
            ollama_kwargs["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            ollama_kwargs["num_predict"] = kwargs["max_new_tokens"]
        base_url = get_ollama_base_url()
        if base_url:
            ollama_kwargs["base_url"] = base_url
        llm = OllamaLLM(model=actual_model_name, **ollama_kwargs)
    else:
        llm = build_text_generation_pipeline(actual_model_name, **kwargs)

    return model_details, llm
