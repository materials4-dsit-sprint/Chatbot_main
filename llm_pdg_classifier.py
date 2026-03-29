"""
llm_pdg_classifier.py

LLM classification pipeline for the LLM phase diagram generator.
- Uses the exact prompt_prefix text and the same robust parsing logic as the original.
- Implements batching, per-batch pause and periodic long break (break_every_batches).
- Writes the authoritative LLM log CSV: OUT_DIR/<safe_formula>_llm_log.csv
- Log columns match the original pipeline so the rest of the code can consume the file.

Dependencies:
- pandas, json, time, os, re
- Expects OUT_DIR and _safe_filename to be provided by llm_phase_diagram_gen (we import them).
"""

from typing import List, Dict, Any, Optional
import os
import io
import time
import csv
import json
import re
import pandas as pd
import time as _time
import concurrent.futures

OUT_DIR = os.path.join("/app/storage", "materials_outputs")

# -------------------------
# Utilities
# -------------------------
def _safe_filename(s: str, maxlen: int = 180) -> str:
    """
    Convert a string into a filesystem-safe filename fragment.

    Parameters
    ----------
    s : str
        Source string to sanitize.
    maxlen : int, optional
        Maximum output length, by default 180.

    Returns
    -------
    str
        Sanitized filename fragment.
    """
    base = re.sub(r"[^\w\-_\.]", "_", s)
    return base[:maxlen]


# -------------------------
# Robust JSON extraction helper
# -------------------------
def _extract_json_from_text(text: str) -> str:
    """
    Extract the JSON-looking portion of a model response.

    Parameters
    ----------
    text : str
        Raw text returned by the model.

    Returns
    -------
    str
        Extracted JSON fragment or the original text.
    """
    m = re.search(r"($begin:math:display$\.\*$end:math:display$)", text, flags=re.S)
    if m:
        return m.group(1)
    m = re.search(r"(\{.*\})", text, flags=re.S)
    if m:
        return m.group(1)
    return text



# -----------------------
# Core LLM invocation helper (adapter point)
# -----------------------
def invoke_llm(prompt: str, llm) -> str:
    """
    Invoke the LLM using the adapter expected by the classifier.

    Parameters
    ----------
    prompt : str
        Prompt text to send to the language model.
    llm : object
        Language model instance implementing `invoke` or callable semantics.

    Returns
    -------
    str
        Raw model response text.
    """
    if llm is None:
        raise RuntimeError("LLM instance required for invoke_llm")
    try:
        if hasattr(llm, "invoke"):
            return llm.invoke(prompt)
        else:
            return llm(prompt)
    except Exception:
        # re-raise so callers can handle; keep behaviour explicit
        raise

def _reset_llm_cache(llm):
    """
    Reset model-side caches when the backend exposes reset hooks.

    Parameters
    ----------
    llm : object
        Language model instance whose state should be cleared.

    Returns
    -------
    None
        This function updates model state in place.
    """
    if llm is None:
        return
    for name in ("reset_state", "reset", "clear_cache", "reload", "invalidate_cache"):
        fn = getattr(llm, name, None)
        if callable(fn):
            try:
                fn()
                print(f"[llm_pdg_classifier] Called llm.{name}() to reset state")
                return
            except Exception as e:
                print(f"[llm_pdg_classifier][WARN] llm.{name}() failed: {e}")
    # last resort: if object has 'client' with reset-like methods, try that
    client = getattr(llm, "client", None)
    if client is not None:
        for name in ("reset_state", "reset", "clear_cache"):
            fn = getattr(client, name, None)
            if callable(fn):
                try:
                    fn()
                    print(f"[llm_pdg_classifier] Called llm.client.{name}() to reset state")
                    return
                except Exception as e:
                    print(f"[llm_pdg_classifier][WARN] llm.client.{name}() failed: {e}")
                    
# -----------------------
# Helper: safe numeric coercion
# -----------------------
def _to_float_or_nan(v):
    """
    Convert a value to float while falling back to NaN on failure.

    Parameters
    ----------
    v : Any
        Value to convert.

    Returns
    -------
    float
        Parsed float value or `nan` when conversion fails.
    """
    try:
        return float(v)
    except Exception:
        return float("nan")
    
# -----------------------
# Defult Prompt
# -----------------------
DEFAULT_PROMPT_TEMPLATE = """You are a careful materials scientist and a strict JSON-output assistant.
Task: For each entry provided, decide whether it belongs to the target compound family described by the formula (A(1-x)B(x), possibly with matrix elements like O or Mn).
Important: I will provide a mapping of letters to element tokens (A,B,C...). Use these letters when reasoning about A and B in expressions like A(1-x)B(x).

Element mapping: {elements_txt}

Input 'Names' values are provided exactly as text; they may contain stoichiometry in forms such as:
- La0.7Sr0.3MnO3
- La 0.7 Sr 0.3 Mn O3
- La0.7 Sr0.3; La0.70Sr0.30
- 70% Sr, 30% La
- (La,Sr)MnO3 or La(1-x)Sr(x)MnO3
- La0.7Sr0.3Mn1-xCrxO3

Guidelines for the model:
- If Names contains explicit numeric fractions for A and B, extract B's fraction as x (La0.7Sr0.3 -> x=0.3).
- If only A's numeric fraction is present, set x = 1 - A.
- If stoichiometry is ambiguous or absent but the row is clearly a member, set include=true and x=null.
- If significant additional cations are present (dopants) at major fractions, mark include=false unless A/B remain clearly the intended pair.
- Output ONLY a single JSON array (one object per input row) with elements exactly:
    - "id": string (must match input row id)
    - "include": true or false
    - "x": number between 0.0 and 1.0, or null
Do not output commentary or extra text.

Target formula: {formula}

Now process the rows below. Return a single JSON array as described.
"""

# -----------------------
# Main classifier (extracted + lightly refactored)
# -----------------------

def classify_rows_with_llm(
    formula: str,
    rows: List[Dict[str, Any]],
    *,
    llm_instance,
    batch_size: int = 3,
    pause_batches: int = 0,
    interactive: bool = False,
    checkpoint_every: Optional[int] = None,
    log_responses: bool = True,
    per_row: bool = False,
    break_every_batches: int = 5,
    break_seconds: int = 10,
    batch_timeout_seconds: int = 60,
    mini_sleep_seconds: float = 1.0,
    prompt_template: Optional[str] = None,
):
    """
    Classify material rows with the configured LLM pipeline.

    Parameters
    ----------
    formula : str
        Target formula used to guide classification.
    rows : List[Dict[str, Any]]
        Candidate rows to classify.
    llm_instance : object
        Language model instance used for evaluation.
    batch_size : int, optional
        Number of rows evaluated per batch, by default 3.
    pause_batches : int, optional
        Pause duration between batches, by default 0.
    interactive : bool, optional
        Whether to run interactively, by default False.
    checkpoint_every : Optional[int], optional
        Interval for optional checkpoints.
    log_responses : bool, optional
        Whether to store raw responses, by default True.
    per_row : bool, optional
        Whether to log each row separately, by default False.
    break_every_batches : int, optional
        Interval for long pauses, by default 5.
    break_seconds : int, optional
        Duration of long pauses in seconds, by default 10.
    batch_timeout_seconds : int, optional
        Timeout applied to each batch invocation, by default 60.
    mini_sleep_seconds : float, optional
        Short sleep between retries or batches, by default 1.0.
    prompt_template : Optional[str], optional
        Prompt template override.

    Returns
    -------
    list
        Parsed classification results for processed rows.
    """
    print(f"[llm_pdg_classifier] classify_rows_with_llm called for formula={formula!r}; total rows={len(rows)}; llm_instance is None? {llm_instance is None}")
    # prepare destinations & filename
    safe = _safe_filename(formula)
    os.makedirs(OUT_DIR, exist_ok=True)
    log_fn = os.path.join(OUT_DIR, f"{safe}_llm_log.csv")

    # Build element mapping snippet used in the original prompt (A,B,C... mapping)
    try:
        from llm_phase_diagram_gen import _extract_element_tokens, _map_tokens_to_letters  # type: ignore
        tokens = _extract_element_tokens(formula)
        mapping = _map_tokens_to_letters(tokens) if tokens else {}
        elements_txt = ", ".join(f"{k}={v}" for k, v in mapping.items()) if mapping else "none detected"
    except Exception:
        elements_txt = "none detected"

#     prompt_prefix = f"""You are a careful materials scientist and a strict JSON-output assistant.
# Task: For each entry provided, decide whether it belongs to the target compound family described by the formula (A(1-x)B(x), possibly with matrix elements like O or Mn).
# Important: I will provide a mapping of letters to element tokens (A,B,C...). Use these letters when reasoning about A and B in expressions like A(1-x)B(x).

# Element mapping: {elements_txt}

# Input 'Names' values are provided exactly as text; they may contain stoichiometry in forms such as:
# - La0.7Sr0.3MnO3
# - La 0.7 Sr 0.3 Mn O3
# - La0.7 Sr0.3; La0.70Sr0.30
# - 70% Sr, 30% La
# - (La,Sr)MnO3 or La(1-x)Sr(x)MnO3
# - La0.7Sr0.3Mn1-xCrxO3

# Guidelines for the model:
# - If Names contains explicit numeric fractions for A and B, extract B's fraction as x (La0.7Sr0.3 -> x=0.3).
# - If only A's numeric fraction is present, set x = 1 - A.
# - If stoichiometry is ambiguous or absent but the row is clearly a member, set include=true and x=null.
# - If significant additional cations are present (dopants) at major fractions, mark include=false unless A/B remain clearly the intended pair.
# - Output ONLY a single JSON array (one object per input row) with elements exactly:
#     - "id": string (must match input row id)
#     - "include": true or false
#     - "x": number between 0.0 and 1.0, or null
# Do not output commentary or extra text.

# Target formula: {formula}

# Now process the rows below. Return a single JSON array as described.
# """
    
    if not prompt_template:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    prompt_prefix = prompt_template.format(
                    formula=formula,
                    elements_txt=elements_txt
                    )
    
    # ensure rows shape & defaults — keep original 'id' but also include authoritative '_id' and DOI
    safe_rows: List[Dict[str, Any]] = []
    for r in rows:
        incoming_id = r.get("id") or str(len(safe_rows))
        safe_rows.append({
            "id": incoming_id,                      # original id field (kept)
            "_id": r.get("_id", incoming_id),       # authoritative _id (fall back to incoming id)
            "DOI": r.get("DOI", None),              # preserve DOI if provided
            "Names": r.get("Names", ""),
            "Type": r.get("Type", ""),
            "Normalised Value": r.get("Normalised Value", None),
        })

    total = len(safe_rows)
    if total == 0:
        return []

        # --- Ensure we have an LLM instance: try fallback from llm_phase_diagram_gen module ---
    if llm_instance is None:
        try:
            # prefer the _llm created by llm_phase_diagram_gen.init_services()
            from llm_phase_diagram_gen import _llm as _fallback_llm  # type: ignore
            if _fallback_llm is not None:
                llm_instance = _fallback_llm
                print("[llm_pdg_classifier] Using fallback _llm imported from llm_phase_diagram_gen")
        except Exception:
            pass

    # If still None, raise (fail fast so we don't silently produce empty results)
    if llm_instance is None:
        raise RuntimeError("classify_rows_with_llm: llm_instance is None. Provide an llm_instance (e.g., classifier_options['llm_instance']=llm_phase_diagram_gen._llm).")

    # Prepare CSV header & writer (append mode: if file exists we append; the original behaviour appended)
    write_header = not os.path.exists(log_fn)
    # include _id and DOI in the authoritative log
    fieldnames = [
        "batch_index", "row_index_in_batch", "global_row_index",
        "id", "_id", "DOI",
        "Names", "Type", "Normalised Value", "prompt", "raw_response",
        "parsed_for_item", "parsed_include", "parsed_x", "timestamp"
    ]

    batch_count = 0
    processed_rows = 0
    decisions_map: Dict[str, Dict[str, Any]] = {}

    # process in batches
    with open(log_fn, "a", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for start in range(0, total, batch_size):
            batch = safe_rows[start:start + batch_size]
            batch_count += 1

            # prepare batch payload exactly as original (list of dicts)
            batch_payload = []
            for r in batch:
                nv = r["Normalised Value"]
                try:
                    nv = float(nv) if nv is not None else None
                except Exception:
                    nv = None
                batch_payload.append({
                    "id": r["id"],
                    "Names": r["Names"],
                    # "Type": r["Type"],
                    # "Normalised Value": nv,
                })

            prompt = prompt_prefix + "\nRows:\n" + json.dumps(batch_payload, ensure_ascii=False)

            # call LLM via adapter but enforce a timeout so a stuck request doesn't hang the whole run
            raw = None
            try:
                print(f"[llm_pdg_classifier] About to invoke LLM for batch {batch_count} (rows {len(batch)})")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(invoke_llm, prompt, llm_instance)
                    try:
                        t0 = time.time()
                        raw = fut.result(timeout=batch_timeout_seconds)
                        t1 = time.time()
                        print(f"[llm_pdg_classifier] LLM invoked for batch {batch_count}; duration={t1-t0:.3f}s; response_len={len(str(raw)) if raw is not None else 0}")
                    except concurrent.futures.TimeoutError:
                        # Timeout: cancel and mark batch as failed (do not block)
                        fut.cancel()
                        print(f"[llm_pdg_classifier][WARN] LLM batch {batch_count} timed out after {batch_timeout_seconds}s; marking rows excluded and continuing")
                        raw = None
            except Exception as e:
                print(f"[llm_pdg_classifier][ERROR] Exception while invoking LLM for batch {batch_count}: {e}")
                raw = None

            # if raw is None treat as LLM failure and mark rows as excluded (same as earlier error path)
            if raw is None:
                for i, r in enumerate(batch):
                    pid = r["id"]
                    writer.writerow({
                        "batch_index": batch_count,
                        "row_index_in_batch": i,
                        "global_row_index": start + i,
                        "id": pid,
                        "_id": r.get("_id", pid),
                        "DOI": r.get("DOI", None),
                        "Names": r["Names"],
                        "Type": r["Type"],
                        "Normalised Value": r["Normalised Value"],
                        "prompt": prompt,
                        "raw_response": None,
                        "parsed_for_item": None,
                        "parsed_include": False,
                        "parsed_x": None,
                        "timestamp": time.time(),
                    })
                    decisions_map[str(r.get("_id", pid))] = {"_id": r.get("_id", pid), "include": False, "x": None}
                processed_rows += len(batch)
            else:
                # existing parsing logic for raw -> parsed_list continues here (unchanged)
                txt = raw if isinstance(raw, str) else str(raw)
                candidate = _extract_json_from_text(txt)
            

            # robust parsing into list as original
            parsed_list: List[Any] = []
            try:
                parsed = json.loads(candidate)
                parsed_list = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                objs = re.findall(r"\{.*?\}", txt, flags=re.S)
                if objs:
                    for o in objs:
                        try:
                            parsed_list.append(json.loads(o))
                        except Exception:
                            parsed_list.append(None)
                else:
                    # fallback line-by-line attempt
                    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                    for ln in lines:
                        try:
                            parsed_list.append(json.loads(ln))
                        except Exception:
                            s = re.sub(r"[,]+$", "", ln)
                            try:
                                parsed_list.append(int(s))
                            except Exception:
                                try:
                                    parsed_list.append(float(s))
                                except Exception:
                                    parsed_list.append(ln)

            # map parsed items back to rows in the batch (index-based mapping as original)
            for i, r in enumerate(batch):
                parsed_item = parsed_list[i] if i < len(parsed_list) else None
                pid = r["id"]

                # defaults
                parsed_include = False
                parsed_x = None

                if isinstance(parsed_item, dict):
                    # if dict contains id we try to match; otherwise take fields
                    try:
                        if parsed_item.get("id") is not None and str(parsed_item.get("id")) != str(pid):
                            # attempt to find matching parsed dict by id
                            found = None
                            for p in parsed_list:
                                if isinstance(p, dict) and str(p.get("id")) == str(pid):
                                    found = p
                                    break
                            if found is not None:
                                parsed_item = found
                    except Exception:
                        pass

                    try:
                        parsed_include = bool(parsed_item.get("include", False))
                        parsed_x = parsed_item.get("x", None)
                    except Exception:
                        parsed_include = False
                        parsed_x = None

                elif parsed_item is None:
                    parsed_include = False
                    parsed_x = None

                else:
                    # primitive mapping (number => x, boolean => include)
                    if isinstance(parsed_item, bool):
                        parsed_include = bool(parsed_item)
                        parsed_x = None
                    elif isinstance(parsed_item, (int, float)):
                        v = float(parsed_item)
                        if 0.0 <= v <= 1.0:
                            parsed_include = True
                            parsed_x = v
                        else:
                            parsed_include = False
                            parsed_x = None
                    elif isinstance(parsed_item, str):
                        low = parsed_item.strip().lower()
                        if low in ("true", "yes", "y", "1"):
                            parsed_include = True
                            parsed_x = None
                        elif low in ("false", "no", "n", "0"):
                            parsed_include = False
                            parsed_x = None
                        else:
                            try:
                                v = float(re.sub(r"[^\d\.\-eE]+", "", low))
                                if 0.0 <= v <= 1.0:
                                    parsed_include = True
                                    parsed_x = v
                                else:
                                    parsed_include = False
                                    parsed_x = None
                            except Exception:
                                parsed_include = False
                                parsed_x = None
                    else:
                        parsed_include = False
                        parsed_x = None

                # coerce
                if isinstance(parsed_include, str):
                    parsed_include = parsed_include.lower() in ("true", "yes", "y", "1")
                else:
                    parsed_include = bool(parsed_include)

                try:
                    if parsed_x is not None:
                        parsed_x = float(parsed_x)
                        if not (0.0 <= parsed_x <= 1.0):
                            parsed_x = None
                except Exception:
                    parsed_x = None

                # save decision keyed by _id (authoritative)
                decisions_map[str(r.get("_id", pid))] = {"_id": r.get("_id", pid), "include": parsed_include, "x": parsed_x}

                # logging row (include _id and DOI)
                if log_responses:
                    writer.writerow({
                        "batch_index": batch_count,
                        "row_index_in_batch": i,
                        "global_row_index": start + i,
                        "id": pid,
                        "_id": r.get("_id", pid),
                        "DOI": r.get("DOI", None),
                        "Names": r["Names"],
                        "Type": r["Type"],
                        "Normalised Value": r["Normalised Value"],
                        "prompt": prompt,
                        "raw_response": txt,
                        "parsed_for_item": parsed_item,
                        "parsed_include": parsed_include,
                        "parsed_x": parsed_x,
                        "timestamp": time.time(),
                    })

            processed_rows += len(batch)
            
            # short pause after each batch to avoid hitting rate limits / transient stalls
            try:
                _reset_llm_cache(llm_instance)
            except Exception:
                pass
            try:
                time.sleep(mini_sleep_seconds)
            except Exception:
                pass
            
            # informational print (same as original)
            print(f"[materials] Completed batch {batch_count}; processed rows up to index {start + len(batch) - 1} (total processed {processed_rows})")

            # periodic long break (exact original behaviour)
            if break_every_batches and break_every_batches > 0 and (batch_count % break_every_batches == 0):
                print(f"[materials] Reached {batch_count} batches — taking a compute break for {break_seconds} seconds...")
                try:
                    time.sleep(break_seconds)
                except Exception as e:
                    print(f"[materials][WARN] Sleep interrupted: {e}")
                print(f"[materials] Resuming after break.")

            # checkpointing behaviour (preserving original minimal checkpointing)
            if checkpoint_every and checkpoint_every > 0 and (batch_count % checkpoint_every == 0):
                cp_name = f"{safe}_checkpoint_batch{batch_count}.csv"
                cp_path = os.path.join(OUT_DIR, cp_name)
                # Note: original created decisions_partial; here we write a snapshot of all decisions so far
                try:
                    pd.DataFrame(list(decisions_map.values())).to_csv(cp_path, index=False)
                    print(f"[materials] Wrote checkpoint -> {cp_path}")
                except Exception as e:
                    print(f"[materials][WARN] Failed to write checkpoint: {e}")

            # pause_batches interactive pause (preserved minimal behaviour)
            if pause_batches and pause_batches > 0 and (batch_count % pause_batches == 0):
                cp_name = f"{safe}_checkpoint_batch{batch_count}.csv"
                cp_path = os.path.join(OUT_DIR, cp_name)
                try:
                    pd.DataFrame(list(decisions_map.values())).to_csv(cp_path, index=False)
                    print(f"[materials] Checkpoint saved to {cp_path} after batch {batch_count}.")
                except Exception:
                    pass

                if interactive and os.isatty(0):
                    while True:
                        try:
                            resp = input(f"[materials] Paused after {batch_count} batches (processed {processed_rows} rows). Continue? (y/n): ").strip().lower()
                        except EOFError:
                            resp = "n"
                        if resp in ("y", "yes"):
                            print("[materials] Continuing after user confirmation.")
                            break
                        elif resp in ("n", "no"):
                            print("[materials] Stopping per user request.")
                            # finalize log (already appended row-by-row); return decisions list
                            out = [decisions_map.get(str(r["_id"]), {"_id": str(r.get("_id", r["id"])), "include": False, "x": None}) for r in safe_rows]
                            return out
                        else:
                            print("Please answer 'y' or 'n'.")

    # assemble final decisions in original order keyed by _id
    decisions_out: List[Dict[str, Any]] = []
    for r in safe_rows:
        decisions_out.append(decisions_map.get(str(r["_id"]), {"_id": str(r.get("_id", r["id"])), "include": False, "x": None}))

    print(f"[materials] classify_rows_with_llm: returning {len(decisions_out)} decisions (log: {log_fn})")
    return decisions_out
