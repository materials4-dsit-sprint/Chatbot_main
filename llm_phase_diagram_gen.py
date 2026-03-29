"""
llm_phase_diagram_gen.py

Light orchestration module: produce JSON-safe Curie / Neel lists for frontend.

Behavior:
- Uses existing prefilter + ranking helpers from the LLM phase diagram generator flow
- Calls llm_pdg_classifier.classify_rows_with_llm(...) to produce/update the LLM log
- Reads the authoritative LLM log (OUT_DIR/<safe>_llm_log.csv) and builds 'curie' and 'neel' lists
  containing dicts with fields x, T, Name (and preserves _id/DOI if present).
- Returns a dict {'curie': [...], 'neel': [...], 'meta': {...}} ready to be returned via JSON.

This module intentionally avoids plotting and binning — frontend handles that.
"""

from typing import Dict, Any, Optional
import os
import json
import numpy as np
import pandas as pd
import re
from cb_embeddings import get_embeddings_provider
from fastapi import APIRouter
from typing import List, Dict, Any, Optional
from helper_llm_runtime import build_llm, get_active_pipeline, get_configured_default_model, resolve_model_selection
router = APIRouter()

from llm_pdg_classifier import classify_rows_with_llm, OUT_DIR, _safe_filename

DEFAULT_MATERIALS_CSV = os.path.join("/app/storage", "materials", "_new_curie_neel_database_processed_cleaned.csv",)
RAW_CSV = os.path.join("/app/storage", "materials", "materials_cleaned_shortened_names_as_they_are_FULL.csv",)
VS_BASE_DIR = os.path.join("/app/storage", "csv_vectorstores",)

# Globals (lazy init)
_embeddings = None
_llm = None
_vs = None
_df = None
_LLM_CACHE: dict[str, object] = {}

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def _sanitize_df_for_json(df: pd.DataFrame) -> list:
    """
    Convert a dataframe into a JSON-safe list of records.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to serialize.

    Returns
    -------
    list
        JSON-safe list of record dictionaries.
    """
    if df is None or df.empty:
        return []
    df2 = df.copy()
    for c in ("x", "T"):
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    txt = df2.to_json(orient="records", date_format="iso", double_precision=10)
    return json.loads(txt)


def _build_phase_data_from_log_df(log_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build Curie and Neel plotting dataframes from the classifier log.

    Parameters
    ----------
    log_df : pandas.DataFrame
        Classifier log dataframe.

    Returns
    -------
    Dict[str, pandas.DataFrame]
        Mapping containing `"curie"` and `"neel"` dataframes.
    """
    if log_df is None or log_df.empty:
        return {"curie": pd.DataFrame(), "neel": pd.DataFrame()}

    df = log_df.copy()

    # standardize include flag
    if "parsed_include" in df.columns:
        df["parsed_include"] = df["parsed_include"].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])
    else:
        df["parsed_include"] = True

    df = df[df["parsed_include"] == True]

    df["parsed_x"] = pd.to_numeric(df.get("parsed_x"), errors="coerce")
    df["Normalised Value"] = pd.to_numeric(df.get("Normalised Value"), errors="coerce")
    df = df.dropna(subset=["parsed_x", "Normalised Value"])
    df = df.rename(columns={"parsed_x": "x", "Normalised Value": "T", "Names": "Name"})

    df["Type_norm"] = df["Type"].astype(str).str.strip().str.lower()

    curie_df = df[df["Type_norm"].str.contains("curie", na=False)][["x", "T", "Name", "_id", "DOI"]].reset_index(drop=True)
    neel_df = df[df["Type_norm"].str.contains("neel|néel", na=False)][["x", "T", "Name", "_id", "DOI"]].reset_index(drop=True)

    return {"curie": curie_df, "neel": neel_df}

def normalize_names(value: Any) -> str:
    """
    Normalize a material name value into a canonical string.

    Parameters
    ----------
    value : Any
        Raw value to normalize.

    Returns
    -------
    str
        Canonicalized name string.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(x).strip() for x in value) if len(value) > 1 else str(value[0]).strip()
    if not isinstance(value, str):
        return str(value).strip()
    s = value.strip()
    if s.startswith("[") or s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return "; ".join(str(x).strip() for x in obj) if obj else ""
            if isinstance(obj, dict):
                return "; ".join(f"{k}:{v}" for k, v in obj.items())
        except Exception:
            pass
    s2 = re.sub(r"^$begin:math:display$\|$end:math:display$$|^'|'$|^\"|\"$", "", s).strip()
    return s2.strip(' \'"')

def _extract_element_tokens(formula: str) -> List[str]:
    """
    Extract formula-like element tokens from a material expression.

    Parameters
    ----------
    formula : str
        Formula string to tokenize.

    Returns
    -------
    List[str]
        Ordered list of extracted element tokens.
    """
    if not formula:
        return []

    # Remove math clutter but keep alphanumeric blocks
    cleaned = re.sub(r"[^\w]", " ", formula)

    # Capture full formula-like chunks (e.g., MnO3, Fe2O4, La)
    tokens = re.findall(r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*", cleaned)

    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)

    return out

# -------------------------
# Embedding helpers
# -------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    a : numpy.ndarray
        First vector.
    b : numpy.ndarray
        Second vector.

    Returns
    -------
    float
        Cosine similarity score.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _embed_query_and_docs(query: str, texts: List[str]) -> (np.ndarray, List[np.ndarray]):
    """
    Embed the query and candidate texts using the active embeddings provider.

    Parameters
    ----------
    query : str
        Query text to embed.
    texts : List[str]
        Candidate row texts to embed.

    Returns
    -------
    tuple
        Query vector and list of document vectors.
    """
    global _embeddings
    if _embeddings is None:
        raise RuntimeError("Embeddings provider not initialized; call init_services() first.")

    try:
        if hasattr(_embeddings, "embed_query"):
            qvec = _embeddings.embed_query(query)
        else:
            qvec = _embeddings.embed_documents([query])[0]
    except Exception:
        qvec = _embeddings.embed_documents([query])[0]

    doc_vecs = []
    try:
        if hasattr(_embeddings, "embed_documents"):
            doc_vecs = _embeddings.embed_documents(texts)
        else:
            for t in texts:
                doc_vecs.append(_embeddings.embed_documents([t])[0])
    except Exception:
        doc_vecs = []
        for t in texts:
            try:
                doc_vecs.append(_embeddings.embed_query(t))
            except Exception:
                doc_vecs.append(_embeddings.embed_documents([t])[0])
    return np.asarray(qvec, dtype=float), [np.asarray(v, dtype=float) for v in doc_vecs]

# -------------------------
# CSV loader
# -------------------------
def load_csv(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and normalize the materials CSV used by the LLM pipeline.

    Parameters
    ----------
    csv_path : Optional[str], optional
        Explicit CSV path override.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe ready for filtering and ranking.
    """
    path = csv_path or RAW_CSV
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = {"Names", "Type", "Normalised Value", "_id"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns: {missing}")

    df["Names"] = df["Names"].apply(normalize_names)
    df["Normalised Value"] = pd.to_numeric(df["Normalised Value"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Names", "Normalised Value"])
    dropped = before - len(df)
    print(f"[materials] Loaded {len(df)} rows (dropped {dropped} incomplete rows).")
    return df

# -------------------------
# Initialization
# -------------------------
def init_services(csv_path: Optional[str] = None):
    """
    Initialize cached embeddings, model, and dataframe state.

    Parameters
    ----------
    csv_path : Optional[str], optional
        Explicit CSV path override.

    Returns
    -------
    None
        This function initializes module-level services in place.
    """
    global _embeddings, _llm, _df, _vs

    if _embeddings is not None and _llm is not None and _df is not None:
        return

    print("[materials] init_services: loading CSV and initializing providers...")
    _df = load_csv(csv_path)

    print("[materials] Initializing embeddings provider...")
    _embeddings = get_embeddings_provider()

    default_model = get_configured_default_model(
        hf_env_vars=("MATERIALS_HF_MODEL", "HF_MODEL"),
        ollama_env_vars=("MATERIALS_OLLAMA_MODEL", "OLLAMA_MODEL"),
    )
    print(
        f"[materials] Initializing {get_active_pipeline()} LLM: {default_model['actual_model_name']}",
    )
    _resolved, _llm = build_llm(
        str(default_model["model_key"] or default_model["actual_model_name"]),
        hf_env_vars=("MATERIALS_HF_MODEL", "HF_MODEL"),
        ollama_env_vars=("MATERIALS_OLLAMA_MODEL", "OLLAMA_MODEL"),
    )
    _LLM_CACHE[str(default_model["actual_model_name"])] = _llm

    vs_dir = os.path.join(VS_BASE_DIR, _safe_filename(os.path.basename(csv_path or RAW_CSV)))
    try:
        _vs = FAISS.load_local(vs_dir, _embeddings, allow_dangerous_deserialization=True)
        print(f"[materials] Loaded existing FAISS from {vs_dir}")
    except Exception:
        _vs = None
        print("[materials] No persistent FAISS loaded (optional).")


def get_materials_llm_instance(selected_model: str | None = None):
    """
    Get or create the materials LLM instance for classification.

    Parameters
    ----------
    selected_model : str | None, optional
        Explicit model selection override.

    Returns
    -------
    tuple
        Resolved model metadata and the instantiated LLM object.
    """
    global _llm

    if selected_model:
        model_details = resolve_model_selection(selected_model, strict=False)
    else:
        model_details = get_configured_default_model(
            hf_env_vars=("MATERIALS_HF_MODEL", "HF_MODEL"),
            ollama_env_vars=("MATERIALS_OLLAMA_MODEL", "OLLAMA_MODEL"),
        )
    actual_model_name = str(model_details["actual_model_name"])

    if _llm is not None and actual_model_name in _LLM_CACHE:
        return model_details, _LLM_CACHE[actual_model_name]

    if _llm is not None and selected_model is None:
        default_model = get_configured_default_model(
            hf_env_vars=("MATERIALS_HF_MODEL", "HF_MODEL"),
            ollama_env_vars=("MATERIALS_OLLAMA_MODEL", "OLLAMA_MODEL"),
        )
        if actual_model_name == str(default_model["actual_model_name"]):
            _LLM_CACHE[actual_model_name] = _llm
            return model_details, _llm

    _resolved, llm_instance = build_llm(
        selected_model,
        hf_env_vars=("MATERIALS_HF_MODEL", "HF_MODEL"),
        ollama_env_vars=("MATERIALS_OLLAMA_MODEL", "OLLAMA_MODEL"),
    )
    _LLM_CACHE[actual_model_name] = llm_instance
    return model_details, llm_instance


# -------------------------
# Prefilter
# -------------------------
def prefilter_by_formula_tokens(formula: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Prefilter rows whose names match all tokens extracted from the formula.

    Parameters
    ----------
    formula : str
        Target material formula.
    csv_path : Optional[str], optional
        Explicit CSV path override.

    Returns
    -------
    pandas.DataFrame
        Prefiltered candidate dataframe.
    """
    init_services(csv_path)
    tokens = _extract_element_tokens(formula)
    print(f"[materials] prefilter tokens extracted: {tokens}")

    if not tokens:
        df_pref = _df.copy()
    else:
        lowers = _df.apply(lambda r: f"{r['Names']} {r['Type']}".lower(), axis=1)
        mask = pd.Series(True, index=_df.index)
        for t in tokens:
            lt = t.lower()
            mask = mask & lowers.str.contains(re.escape(lt))
        df_pref = _df[mask].copy()

    safe = _safe_filename(formula)
    pref_name = f"{safe}_prefilter.csv"
    pref_path = os.path.join(OUT_DIR, pref_name)
    df_pref.to_csv(pref_path, index=False)
    print(f"[materials] Wrote prefilter CSV: {pref_path} (rows: {len(df_pref)})")
    return df_pref

# -------------------------
# Rank prefiltered rows by embedding similarity (in-memory)
# -------------------------
def rank_prefiltered_rows_by_similarity(formula: str, df_pref: pd.DataFrame, k: int = 200) -> pd.DataFrame:
    """
    Rank candidate rows by embedding similarity to the target formula.

    Parameters
    ----------
    formula : str
        Target material formula.
    df_pref : pandas.DataFrame
        Prefiltered candidate dataframe.
    k : int, optional
        Maximum number of rows to keep, by default 200.

    Returns
    -------
    pandas.DataFrame
        Ranked dataframe of top candidate rows.
    """
    texts = []
    for _, r in df_pref.iterrows():
        texts.append(f"{r['Names']} | {r['Type']} | {r['Normalised Value']}")

    qvec, doc_vecs = _embed_query_and_docs(formula, texts)
    scores = [_cosine(qvec, dv) for dv in doc_vecs]

    dfc = df_pref.copy().reset_index(drop=True)
    dfc["score"] = scores
    dfc = dfc.sort_values("score", ascending=False).reset_index(drop=True)
    if k is not None and k > 0:
        dfc_top = dfc.head(k).copy()
    else:
        dfc_top = dfc.copy()

    safe = _safe_filename(formula)
    cand_name = f"{safe}_candidates.csv"
    cand_path = os.path.join(OUT_DIR, cand_name)
    dfc_top.to_csv(cand_path, index=False)
    print(f"[materials] Wrote candidates CSV: {cand_path} (rows: {len(dfc_top)})")
    return dfc_top

# ---------------------------------------------------
# Core function
# ---------------------------------------------------
def build_material_phase_data(
    formula: str,
    csv_path: Optional[str] = DEFAULT_MATERIALS_CSV,
    log_mode: str = "append",
    classifier_options: Optional[Dict[str, Any]] = None,
    # prompt_template: str,
) -> Dict[str, Any]:
    """
    Build phase data by running the full LLM classification workflow.

    Parameters
    ----------
    formula : str
        Target material formula.
    csv_path : Optional[str], optional
        Explicit CSV path override.
    log_mode : str, optional
        Log handling mode: `"use"`, `"append"`, or `"recompute"`.
    classifier_options : Optional[Dict[str, Any]], optional
        Extra options forwarded to the classifier.

    Returns
    -------
    Dict[str, Any]
        JSON-ready Curie and Neel data with metadata.
    """

    if log_mode not in ("use", "append", "recompute"):
        log_mode = "append"

    safe = _safe_filename(formula)
    log_fn = os.path.join(OUT_DIR, f"{safe}_llm_log.csv")

    # DEBUG: show what we're about to do (helpful for troubleshooting recompute)
    print(f"[materials] build_material_phase_data called with formula={formula!r}, safe={safe!r}, log_mode={log_mode!r}")
    print(f"[materials] expected log path: {log_fn}")

    # Ensure CSV exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Fast-path: use existing log
    if log_mode == "use":
        if os.path.exists(log_fn):
            try:
                df_log = pd.read_csv(log_fn, encoding="utf-8", low_memory=False)
                phase = _build_phase_data_from_log_df(df_log)
                return {
                    "curie": _sanitize_df_for_json(phase["curie"]),
                    "neel": _sanitize_df_for_json(phase["neel"]),
                    "meta": {"log_mode": "use", "log_path": log_fn},
                }
            except Exception as e:
                return {"error": f"Failed to read existing log: {e}"}
        else:
            return {"error": "No existing LLM log (use mode requested but log missing)."}

    # Ensure recompute truly deletes the log before anything else
    if log_mode == "recompute":
        if os.path.exists(log_fn):
            try:
                os.remove(log_fn)
                print(f"[materials] recompute: removed existing log: {log_fn}")
            except Exception as e:
                print(f"[materials][WARN] recompute requested but failed to remove {log_fn}: {e}")
        else:
            print(f"[materials] recompute: no existing log to remove at {log_fn}")

    init_services(csv_path)  # ensures embeddings/llm/load CSV as original did

    df_pref = prefilter_by_formula_tokens(formula, csv_path)
    # reasonable default candidate size preserved from original pipeline
    k_eff = min(400, len(df_pref)) if len(df_pref) > 0 else 0
    df_candidates = rank_prefiltered_rows_by_similarity(formula, df_pref, k=k_eff)

    # If append and existing log present, drop those ids from candidates (original behaviour)
    if log_mode == "append" and os.path.exists(log_fn):
        try:
            existing_df = pd.read_csv(log_fn, encoding="utf-8", low_memory=False)
            if "id" in existing_df.columns:
                processed_ids = set(existing_df["id"].astype(str).tolist())
                df_candidates = df_candidates[~df_candidates["_id"].astype(str).isin(processed_ids)].reset_index(drop=True)
            else:
                # if no id column, treat as no processed ids
                pass
        except Exception:
            pass

    # prepare rows_for_llm
    rows_for_llm = []
    for _, r in df_candidates.iterrows():
        rid = str(r["_id"])
        rows_for_llm.append({
            "id": rid,
            "Names": r["Names"],
            "Type": r["Type"],
            "Normalised Value": r["Normalised Value"],
        })
    
    # If recompute requested but no candidates were found, create an empty authoritative log
    if log_mode == "recompute" and not rows_for_llm:
        # touch an empty log file with header to mark recompute was done
        try:
            header = ["batch_index", "row_index_in_batch", "global_row_index", "id", "Names", "Type", "Normalised Value", "prompt", "raw_response", "parsed_for_item", "parsed_include", "parsed_x"]
            with open(log_fn, "w", encoding="utf-8", newline="") as fh:
                import csv
                writer = csv.writer(fh)
                writer.writerow(header)
            print(f"[materials] recompute: created empty authoritative log at {log_fn} (no candidates).")
        except Exception as e:
            print(f"[materials][WARN] failed to create empty log at {log_fn}: {e}")
        # and then return empty result immediately (no classifier call)
        return {"curie": [], "neel": [], "meta": {"log_mode": log_mode, "log_path": log_fn, "note": "recompute with zero candidates"}}
    
    # if there are rows to classify, call the classifier
    classifier_options = classifier_options or {}
    if rows_for_llm:
        # Ensure init_services was run so llm_phase_diagram_gen._llm exists
        try:
            init_services(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize services before classification: {e}")

        try:
            _model_details, classifier_options["llm_instance"] = get_materials_llm_instance(
                classifier_options.get("model"),
            )
        except Exception as e:
            raise RuntimeError(f"llm_instance required but llm_phase_diagram_gen LLM unavailable: {e}")

        # Call classifier with provided or default options
        classify_rows_with_llm(
            formula,
            rows_for_llm,
            llm_instance=classifier_options.get("llm_instance"),
            batch_size=classifier_options.get("batch_size", 3),
            pause_batches=classifier_options.get("pause_batches", 0),
            interactive=classifier_options.get("interactive", False),
            checkpoint_every=classifier_options.get("checkpoint_every", None),
            log_responses=classifier_options.get("log_responses", True),
            per_row=classifier_options.get("per_row", False),
            break_every_batches=classifier_options.get("break_every_batches", 5),
            break_seconds=classifier_options.get("break_seconds", 10),
            prompt_template=classifier_options.get("prompt_template"),
        )

    # After classifier runs, read the authoritative log and build phase data
    if not os.path.exists(log_fn):
        return {"curie": [], "neel": [], "meta": {"warning": "Log not created"}}

    try:
        df_log = pd.read_csv(log_fn, encoding="utf-8", low_memory=False)
    except Exception as e:
        return {"error": f"Failed to read log after classification: {e}"}

    phase = _build_phase_data_from_log_df(df_log)
    return {
        "curie": _sanitize_df_for_json(phase["curie"]),
        "neel": _sanitize_df_for_json(phase["neel"]),
        "meta": {"log_mode": log_mode, "log_path": log_fn, "candidates_count": len(rows_for_llm)},
    }

@router.post("/llm_phase_gen")
def llm_phase_gen_endpoint(
    formula: str,
    log_mode: str = "append",
    prompt_template: Optional[str] = None,
    model: Optional[str] = None,
):
    """
    Serve LLM-based phase diagram data through the FastAPI router.

    Parameters
    ----------
    formula : str
        Target material formula.
    log_mode : str, optional
        Log handling mode for the classifier.
    prompt_template : Optional[str], optional
        Prompt template override.
    model : Optional[str], optional
        Model selection override.

    Returns
    -------
    Dict[str, Any]
        JSON response payload for the frontend.
    """
    classifier_options = {}

    if prompt_template:
        classifier_options["prompt_template"] = prompt_template
    if model:
        classifier_options["model"] = model

    data = build_material_phase_data(
        formula=formula,
        log_mode=log_mode,
        classifier_options=classifier_options,
    )
    return data
