#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frontend_app.py

Final frontend that:
- Uses a chat tab for the chatbot
- Adds two Phase Diagram tabs (LLM and no-LLM) using hvPlot/HoloViews
- Shared interactive widgets: Scatter/Errorbar, Mean/Median, hover shows DOI
- Caches data so toggles update the plot without re-fetching
"""

import os
import asyncio
import json
import requests
import panel as pn
import pandas as pd
import holoviews as hv
hv.extension("bokeh")
import hvplot.pandas  # activates hvplot
from collections import OrderedDict
from llm_runtime import get_model_options, get_default_model_key
pn.config.sizing_mode = "stretch_width"
pn.extension("filedropper")


# --------------------
# Configuration
# --------------------
API_KEY = os.environ.get("API_KEY")          # must match backend

ENDPOINT = os.environ.get("ENDPOINT", "http://localhost:9000/generate")
STREAM_ENDPOINT = os.environ.get("STREAM_ENDPOINT", "http://localhost:9000/generate-stream")
UPLOAD_ENDPOINT = os.environ.get("UPLOAD_ENDPOINT", "http://localhost:9000/upload-context")
PHASE_GEN_ENDPOINT = os.environ.get("PHASE_GEN_ENDPOINT", "http://127.0.0.1:9000/phase_gen")
PHASE_MATERIALS_ENDPOINT = os.environ.get("PHASE_MATERIALS_ENDPOINT", "http://127.0.0.1:9000/materials_phase")


# --------------------
# Login
# --------------------
USERS = {
    "Cam26": "admin",
    "user": "password"
}

username_input = pn.widgets.TextInput(name="Username")
password_input = pn.widgets.PasswordInput(name="Password")
login_message = pn.pane.Markdown("")
login_button = pn.widgets.Button(name="Login", button_type="primary")

login_view = pn.Column(
    pn.pane.Markdown("## Login"),
    username_input,
    password_input,
    login_button,
    login_message,
    width=300,
    align="center"
)


def authenticate(event):
    user = username_input.value
    pwd = password_input.value

    if user in USERS and USERS[user] == pwd:
        content[:] = [tabs]
        print("LOGIN CLICKED")
    else:
        login_message.object = "❌ Invalid username or password"


# default timeout in seconds; override with env var if desired
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "360"))




# --------------------
# Chats
# --------------------
def _build_chat_payload(message: str) -> dict:
    return {
        "question": message,
        "k": int(k_slider.value),
        "log": bool(log_toggle.value),
        "model": llm_menu.value,
        "context_source": context_source_selector.value,
    }


def _chat_headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _sync_chat_request(payload: dict) -> str:
    headers = _chat_headers()
    try:
        response = requests.post(
            ENDPOINT,
            json=payload,
            headers=headers,
            timeout=float(timeout_slider.value),
        )
        print(llm_menu.value)
        if response.status_code != 200:
            return f"❌ Error {response.status_code}: {response.text}"
        data = response.json()
        return data.get("text", "<no response>")
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"


def _format_page_label(page) -> str:
    if isinstance(page, int):
        return str(page + 1)
    if isinstance(page, str) and page.isdigit():
        return str(int(page) + 1)
    if page is None:
        return ""
    return str(page)


def _format_retrieved_chunks_message(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""

    rendered_docs = [f"### Retriever output ({len(retrieved)} documents)"]
    for item in retrieved:
        rank = item.get("rank", "?")
        filename = item.get("filename") or os.path.basename(item.get("source") or "") or "PDF"
        page_label = _format_page_label(item.get("page"))
        snippet = item.get("snippet") or "_Empty chunk_"

        summary = f"Document {rank} | {filename}"
        if page_label:
            summary += f" | page {page_label}"

        rendered_docs.append(
            "<details>"
            f"<summary><code>{summary}</code></summary>\n\n"
            f"```text\n{snippet}\n```\n"
            "</details>"
        )

    return "\n\n".join(rendered_docs)


def _collect_pdf_chat_stream(payload: dict) -> tuple[str | None, str]:
    headers = _chat_headers()
    try:
        with requests.post(
            STREAM_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=float(timeout_slider.value),
            stream=True,
        ) as response:
            if response.status_code != 200:
                return None, f"❌ Error {response.status_code}: {response.text}"

            retrieval_message = None
            answer_text = None
            saw_event = False
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                saw_event = True
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    return None, f"❌ Invalid stream response: {raw_line}"

                event_type = event.get("event")
                if event_type == "retrieval":
                    retrieval_message = _format_retrieved_chunks_message(event.get("retrieved", []))
                elif event_type == "answer":
                    answer_text = event.get("text", "<no response>")
                elif event_type == "error":
                    detail = event.get("detail", "Unknown streaming error")
                    status_code = event.get("status_code")
                    prefix = f"❌ Error {status_code}: " if status_code else "❌ "
                    return retrieval_message, prefix + str(detail)

            if not saw_event:
                return retrieval_message, "❌ Empty response from backend."
            if answer_text is None:
                return retrieval_message, "❌ Backend stream ended before returning an answer."
            return retrieval_message, answer_text
    except requests.exceptions.RequestException as e:
        return None, f"❌ Request failed: {e}"


async def chat_callback(message, user, chat):
    payload = _build_chat_payload(message)
    if payload["context_source"] == "pdfs":
        retrieval_message, answer_text = await asyncio.to_thread(_collect_pdf_chat_stream, payload)
        if retrieval_message:
            chat.send(retrieval_message, user="Assistant", respond=False)
        return answer_text

    return await asyncio.to_thread(_sync_chat_request, payload)


def upload_context_file(file_widget, context_source: str, status_pane):
    filename = None
    file_value = None

    if isinstance(file_widget.value, dict):
        if file_widget.value:
            filename, file_value = next(iter(file_widget.value.items()))
            if isinstance(file_value, str):
                file_value = file_value.encode("utf-8")
    else:
        filename = getattr(file_widget, "filename", None)
        file_value = file_widget.value

    if not file_value or not filename:
        status_pane.object = "❌ Select a file first."
        return

    headers = {"Authorization": f"Bearer {API_KEY}"}
    mime_type = "application/pdf" if context_source == "pdfs" else "text/csv"
    files = {"file": (filename, file_value, mime_type)}
    data = {"context_source": context_source}

    status_pane.object = f"Uploading `{filename}` and building the vector store..."

    try:
        response = requests.post(
            UPLOAD_ENDPOINT,
            data=data,
            files=files,
            headers=headers,
            timeout=float(timeout_slider.value),
        )
        if response.status_code != 200:
            status_pane.object = f"❌ Error {response.status_code}: {response.text}"
            return

        payload = response.json()
        source_label = "PDF" if context_source == "pdfs" else "CSV"
        status_pane.object = (
            f"✅ {source_label} uploaded: `{payload.get('filename')}`. "
            f"PDF stores: {payload.get('pdf_vectorstores', 0)} | "
            f"CSV stores: {payload.get('csv_vectorstores', 0)}"
        )
    except requests.exceptions.RequestException as e:
        status_pane.object = f"❌ Upload failed: {e}"


# log_toggle = pn.widgets.Toggle(
#     name="LLM Logs",
#     value=True,
#     button_type="default",
#     button_style="outline",
#     width=120
# )

log_toggle = pn.widgets.Select(
    name="Logs",
    options={
        "On": True,
        "Off": False
    },
    value=True,
    width=120
)

# Discrete sliders for K (number of rows) and request timeout (seconds)
k_slider = pn.widgets.DiscreteSlider(
    name="K (rows)",
    options=[1, 2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80],
    value=30,
    width=260
)

timeout_slider = pn.widgets.DiscreteSlider(
    name="Request timeout (s)",
    options=[120, 240, 360, 480, 600, 720, 840],
    value=360,   # default seconds
    width=260
)

# llm_menu = pn.widgets.MenuButton(
#     name="LLMs",
#     items=["deepseek-r1:8b", "deepseek-r1:7b", "deepseek-r1:1.5b"],
#     value="deepseek-r1:8b",
#     width=120
# )

llm_menu = pn.widgets.Select(
    name="LLM",
    options=get_model_options(),
    value=get_default_model_key(),
    width=160
)

context_source_selector = pn.widgets.RadioButtonGroup(
    name="Search in",
    options={"PDFs": "pdfs", "CSVs": "csvs"},
    value="pdfs",
    button_type="default",
    button_style="outline",
    width=180,
)

FileDropper = getattr(pn.widgets, "FileDropper", pn.widgets.FileInput)

if FileDropper is pn.widgets.FileInput:
    pdf_dropper = FileDropper(name="PDF dropper", accept=".pdf", multiple=False)
    csv_dropper = FileDropper(name="CSV dropper", accept=".csv", multiple=False)
else:
    pdf_dropper = FileDropper(
        name="PDF dropper",
        accepted_filetypes=[".pdf"],
        multiple=False,
        max_file_size="100MB",
        layout="compact",
        sizing_mode="stretch_width",
    )
    csv_dropper = FileDropper(
        name="CSV dropper",
        accepted_filetypes=[".csv", "text/csv"],
        multiple=False,
        max_file_size="100MB",
        layout="compact",
        sizing_mode="stretch_width",
    )

pdf_upload_status = pn.pane.Markdown("", sizing_mode="stretch_width")

csv_upload_status = pn.pane.Markdown("", sizing_mode="stretch_width")

pdf_dropper.param.watch(lambda event: upload_context_file(pdf_dropper, "pdfs", pdf_upload_status) if event.new else None, "value")
csv_dropper.param.watch(lambda event: upload_context_file(csv_dropper, "csvs", csv_upload_status) if event.new else None, "value")

chat = pn.chat.ChatInterface(
    callback=chat_callback,
    user="Cam26",
    sizing_mode="stretch_both",
    min_width=400,
    width_policy="max",
)

chat.send(
    "Hi! Ask me questions about the PDFs/CSVs I have indexed 📄",
    user="Assistant",
    # avatar=''
    respond=False,
)

left_panel = pn.Column(
    chat,
    sizing_mode="stretch_both",
    min_width=400,
    width_policy="max",
)

right_panel = pn.Column(
    pn.Spacer(height=10),
    context_source_selector,
    log_toggle,
    llm_menu,
    k_slider,
    timeout_slider,
    pdf_dropper,
    pdf_upload_status,
    csv_dropper,
    csv_upload_status,
    width=280,
    sizing_mode="fixed",
    styles={"gap": "5px"},
)

chat_panel = pn.Row(
    left_panel,
    right_panel,
    sizing_mode="stretch_both",
    width_policy="max",
    styles={"align-items": "stretch", "gap": "16px"},
)

# -----------------------
# Shared cached data (module-level)
# -----------------------
_cached_NDF = None
_cached_CDF = None

# -----------------------
# Shared widgets (plot control)
# -----------------------
plot_type = pn.widgets.RadioButtonGroup(
    name="Plot Type",
    options=["Scatter", "Errorbar"],
    value="Scatter",
    button_type="default",
    button_style="outline", width=160
)
# center_selector = pn.widgets.Select(name="Center", options=["mean", "median"], value="mean", width=120)
center_selector = pn.widgets.RadioButtonGroup(
    name="Center",
    options=OrderedDict([("Mean","mean"), ("Median", "median")]),
    value="mean",
    button_type="default",
    button_style="outline", width=130
)


# -----------------------
# Helper: Build plot from cached data (reusable)
# -----------------------
def build_plot(plot_pane, control_plot_type, control_center_selector):
    """
    Build the plot from _cached_NDF (Neel) and _cached_CDF (Curie) and set plot_pane.object.
    plot_pane : pn.pane.HoloViews instance
    control_plot_type : widget (RadioButtonGroup)
    control_center_selector : widget (Select)
    """
    global _cached_NDF, _cached_CDF

    # Normalize cached objects -> DataFrames or empty
    NDF = None
    CDF = None
    try:
        if isinstance(_cached_NDF, pd.DataFrame):
            NDF = _cached_NDF
        elif _cached_NDF is not None:
            NDF = pd.DataFrame(_cached_NDF)
    except Exception:
        NDF = pd.DataFrame([])

    try:
        if isinstance(_cached_CDF, pd.DataFrame):
            CDF = _cached_CDF
        elif _cached_CDF is not None:
            CDF = pd.DataFrame(_cached_CDF)
    except Exception:
        CDF = pd.DataFrame([])

    # defensive emptiness
    if (NDF is None or NDF.empty) and (CDF is None or CDF.empty):
        plot_pane.object = None
        return

    # Scatter mode
    if control_plot_type.value == "Scatter":
        plots = []
        if CDF is not None and not CDF.empty:
            pcur = CDF.hvplot.scatter(
                x="x", y="T", label="CURIE", hover_cols=["Name", "DOI"], size=60
            )
            plots.append(pcur)
        if NDF is not None and not NDF.empty:
            pneel = NDF.hvplot.scatter(
                x="x", y="T", label="NEEL", hover_cols=["Name", "DOI"], size=60
            )
            plots.append(pneel)
        final = plots[0] * plots[1] if len(plots) > 1 else (plots[0] if plots else None)

    # Errorbar mode
    else:
        agg_method = control_center_selector.value  # "mean" or "median"
        import holoviews as _hv  # local alias

        def make_agg(df, label):
            if df is None or df.empty:
                return None, None
            # groupby x to compute center and std
            if agg_method == "mean":
                center = df.groupby("x")["T"].mean()
            else:
                center = df.groupby("x")["T"].median()
            std = df.groupby("x")["T"].std(ddof=0).fillna(0)
            agg_df = pd.DataFrame({"x": center.index, "center": center.values, "std": std.values})
            color = "red" if label == "CURIE" else "blue"

            # ErrorBars (x, y, yerr)
            eb = _hv.ErrorBars([(row.x, row.center, row.std) for row in agg_df.itertuples(index=False)], label=label).opts(color=color)
            # Shaded area between curve and x-axis
            area = _hv.Area((agg_df["x"].values, agg_df["center"].values), "x", "center", label=label).opts(alpha=0.15, color=color)
            # Line and markers (line: Curve, markers: Scatter)
            line = _hv.Curve((agg_df["x"].values, agg_df["center"].values), "x", "center", label=label).opts(line_width=2, color=color)
            markers = _hv.Scatter((agg_df["x"].values, agg_df["center"].values), "x", "center", label=label).opts(marker="o", size=6, color=color)
            # return errorbars and the overlay of (area * line * markers)
            return eb, (area * line * markers)

        agg_plots = []
        if CDF is not None and not CDF.empty:
            eb_c, curie_el = make_agg(CDF, "CURIE")
            if eb_c is not None:
                agg_plots.extend([eb_c, curie_el])
        if NDF is not None and not NDF.empty:
            eb_n, neel_el = make_agg(NDF, "NEEL")
            if eb_n is not None:
                agg_plots.extend([eb_n, neel_el])

        final = None
        if agg_plots:
            final = agg_plots[0]
            for el in agg_plots[1:]:
                final = final * el

    # major grid only
    if final is not None:
        final = final.opts(show_grid=True)
        plot_pane.object = final
    else:
        plot_pane.object = None


# -----------------------
# No-LLM tab widgets and pane
# -----------------------
mat_a = pn.widgets.TextInput(name="Material A (1-x)", value="La", width=180)
mat_b = pn.widgets.TextInput(name="Material B (x)", value="Sr", width=180)
mat_c = pn.widgets.TextInput(name="Material C (invariant)", value="MnO3", width=180)
gen_steps = pn.widgets.IntInput(name="Steps", value=101, start=3, width=180)
# gen_log_mode = pn.widgets.Select(name="Log mode", options=["append", "use", "recompute"], value="append", width=140)

gen_button = pn.widgets.Button(name="Generate", button_type="primary")
gen_plot_pane = pn.pane.HoloViews(height=450, width=900)


def fetch_data(event=None):
    """
    No-LLM fetch: call /phase_gen which returns JSON (neel + curie).
    """
    global _cached_NDF, _cached_CDF

    A = mat_a.value.strip()
    B = mat_b.value.strip()
    C = mat_c.value.strip()
    if not (A and B and C):
        gen_plot_pane.object = None
        return pn.state.notifications.error("All three material fields (A, B, C) are required")

    params = {"A": A, "B": B, "C": C, "n_steps": gen_steps.value}
    # if server supports log_mode for this endpoint, include it (optional)
    # params["log_mode"] = gen_log_mode.value if gen_log_mode.value else "append"

    try:
        resp = requests.post(PHASE_GEN_ENDPOINT, params=params, timeout=300, headers={"Authorization": f"Bearer {API_KEY}"})
        if resp.status_code != 200:
            gen_plot_pane.object = None
            return pn.state.notifications.error(f"Error {resp.status_code}: {resp.text}")

        data = resp.json()
        # server returns "neel" and "curie" records
        _cached_NDF = pd.DataFrame(data.get("neel", []))
        _cached_CDF = pd.DataFrame(data.get("curie", []))

        # build initial plot
        build_plot(gen_plot_pane, plot_type, center_selector)

        # show helpful metadata if present
        meta = data.get("meta", {})
        info = []
        if meta.get("log_path"):
            info.append(f"log: {meta.get('log_path')}")
        if meta.get("candidates_count") is not None:
            info.append(f"candidates: {meta.get('candidates_count')}")
        if info:
            if getattr(pn.state, "notifications", None) is not None:
                pn.state.notifications.info("\n".join(info))
            else:
                print("INFO:", "\n".join(info))

    except Exception as e:
        _cached_NDF = None
        _cached_CDF = None
        llm_plot_pane.object = None
        if getattr(pn.state, "notifications", None) is not None:
            pn.state.notifications.error(f"Request failed: {e}")
        else:
            # fallback so you still see the error in server logs
            print("ERROR: Request failed:", e)


gen_button.on_click(fetch_data)

gen_panel = pn.Column(
    "## Phase Diagram Builder (script)",
    pn.Row(mat_a, mat_b, mat_c, gen_steps),
    pn.Spacer(height=10),
    pn.Row(gen_button, pn.Spacer(width=20), plot_type, 
           pn.Spacer(width=20), center_selector, sizing_mode="fixed"),
    # pn.Row(plot_type, pn.Spacer(width=10), center_selector, sizing_mode="fixed"),
    pn.Spacer(height=10),
    gen_plot_pane,
)

# -----------------------
# LLM tab widgets and pane
# -----------------------
llm_mat_a = pn.widgets.TextInput(name="Material A (1-x)", value="La", width=180)
llm_mat_b = pn.widgets.TextInput(name="Material B (x)", value="Sr", width=180)
llm_mat_c = pn.widgets.TextInput(name="Material C (invariant)", value="MnO3", width=180)
llm_steps = pn.widgets.IntInput(name="Steps", value=101, start=3, width=180)
llm_select = pn.widgets.Select(
    name="LLM",
    options=get_model_options(),
    value=get_default_model_key(),
    width=180,
)

llm_log_mode = pn.widgets.RadioButtonGroup(
    name="Log mode",
    options=OrderedDict([("Append","append"), ("Use","use"), ("Recompute","recompute")]),
    value="use",
    button_type="default",
    button_style="outline", width=220
)

# llm_log_mode = pn.widgets.Select(name="Log mode", options=["append", "use", "recompute"], value="append", width=140)

llm_gen_button = pn.widgets.Button(name="Generate", button_type="primary")
llm_plot_pane = pn.pane.HoloViews(height=450, width=900)


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


# Prompt template editable from frontend (default = the existing prompt you use)
prompt_template_input = pn.widgets.TextAreaInput(
    name="Prompt (Do not remove: Element mapping: {elements_txt}, Target formula: {formula})",
    value=DEFAULT_PROMPT_TEMPLATE,
    height=150,
    sizing_mode="stretch_width"
)


def on_llm_fetch(event=None):
    """
    Fetch JSON from /materials_phase, cache results and build plot.
    """
    global _cached_NDF, _cached_CDF

    A = llm_mat_a.value.strip()
    B = llm_mat_b.value.strip()
    C = llm_mat_c.value.strip()
    if not (A and B and C):
        llm_plot_pane.object = None
        return pn.state.notifications.error("All three material fields (A, B, C) are required")

    formula = f"{A}(1-x){B}(x){C}"
    params = {"formula": formula, "log_mode": llm_log_mode.value,
              "prompt_template": prompt_template_input.value,
              "model": llm_select.value,}

    try:
        resp = requests.post(PHASE_MATERIALS_ENDPOINT, params=params, timeout=300, headers={"Authorization": f"Bearer {API_KEY}"})
        if resp.status_code != 200:
            llm_plot_pane.object = None
            return pn.state.notifications.error(f"Error {resp.status_code}: {resp.text}")

        data = resp.json()
        _cached_NDF = pd.DataFrame(data.get("neel", []))
        _cached_CDF = pd.DataFrame(data.get("curie", []))

        build_plot(llm_plot_pane, plot_type, center_selector)

        meta = data.get("meta", {})
        info = []
        if meta.get("log_path"):
            info.append(f"log: {meta.get('log_path')}")
        if meta.get("candidates_count") is not None:
            info.append(f"candidates: {meta.get('candidates_count')}")
        if info:
            if getattr(pn.state, "notifications", None) is not None:
                pn.state.notifications.info("\n".join(info))
            else:
                print("INFO:", "\n".join(info))

    except Exception as e:
        _cached_NDF = None
        _cached_CDF = None
        llm_plot_pane.object = None
        if getattr(pn.state, "notifications", None) is not None:
            pn.state.notifications.error(f"Request failed: {e}")
        else:
            # fallback so you still see the error in server logs
            print("ERROR: Request failed:", e)


llm_gen_button.on_click(on_llm_fetch)

# -----------------------
# Wire interactive watchers so toggles update without re-fetching
# -----------------------
# When plot_type or center_selector change, rebuild both panes from the cached data
plot_type.param.watch(lambda ev: build_plot(gen_plot_pane, plot_type, center_selector), "value")
plot_type.param.watch(lambda ev: build_plot(llm_plot_pane, plot_type, center_selector), "value")
center_selector.param.watch(lambda ev: build_plot(gen_plot_pane, plot_type, center_selector), "value")
center_selector.param.watch(lambda ev: build_plot(llm_plot_pane, plot_type, center_selector), "value")

# -----------------------
# Compose panels and template
# -----------------------
phase_panel = pn.Column(
    "## Phase Diagram Builder (LLM)",
    pn.Row(llm_mat_a, llm_mat_b, llm_mat_c, llm_steps, llm_select),
    prompt_template_input,
    # pn.Row(llm_log_mode),
    pn.Spacer(height=10),
    pn.Row(llm_log_mode, pn.Spacer(width=20),
           llm_gen_button, pn.Spacer(width=20), plot_type, 
           pn.Spacer(width=20), center_selector, sizing_mode="fixed"),
    # pn.Row(plot_type, pn.Spacer(width=10), center_selector, sizing_mode="fixed"),
    pn.Spacer(height=10),
    llm_plot_pane,
)

tabs = pn.Tabs(("Chat", chat_panel),
               ("Phase diagram (LLM)", phase_panel),
               ("Phase diagram (script)", gen_panel), 
               sizing_mode="stretch_both",)


template = pn.template.FastListTemplate(
    title="Materials 4.0 ChatBot",
    logo="/app/storage/logos/hri_cam.png" if os.path.exists("/app/storage/logos/hri_cam.png") else None,
    header_color="black",
    main=[],
    accent="#00BDB6",
)

# Container for dynamic content
content = pn.Column(sizing_mode="stretch_both")
template.main[:] = [content]

login_button.on_click(authenticate)

#Initial view
content[:] = [login_view]
template.servable()
