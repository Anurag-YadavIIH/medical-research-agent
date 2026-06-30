"""Streamlit frontend for the Medical Research Agent.

This file only calls the backend's HTTP API and renders the response — all
synthesis, citation handling, retrieval and evidence grading happen
server-side. The frontend container doesn't install the backend package (see
docker/frontend.Dockerfile), so BACKEND_URL is read from the environment
directly rather than via config.Settings.

Two modes, selected from the sidebar:
- Normal Search: the original one-shot, ungrouped search (unchanged).
- Projects: a Claude-Projects/NotebookLM-style workspace — scoped search
  history, uploaded-PDF papers, and a chat grounded in that project's corpus.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st
from transforms import (
    comparison_matrix_rows,
    comparison_narrative,
    evidence_summary_stats,
    project_document_rows,
    project_history_rows,
    reference_rows,
    study_detail_rows,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DISCLAIMER = (
    "For research and educational purposes only. Clinical decisions should rely "
    "on professional judgment and full-text evidence review."
)
REQUEST_TIMEOUT_SECONDS = 300
UPLOAD_TIMEOUT_SECONDS = 120
MAX_PAPERS_BOUNDS = (1, 50)
ARTICLE_TYPE_OPTIONS = [
    "Randomized Controlled Trial",
    "Meta-Analysis",
    "Systematic Review",
    "Review",
]

st.set_page_config(page_title="Medical Research Agent", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔬 Medical Research Agent")
st.caption("Multi-agent, fully-cited biomedical evidence synthesis — built on PubMed literature.")
st.warning(f"⚠️ {DISCLAIMER}")


# --- Shared helpers ----------------------------------------------------


def _api_request(method: str, path: str, **kwargs: Any) -> requests.Response | None:
    """Call the backend, surfacing any failure via st.error and returning None
    so callers can bail out with ``if resp is None: return``.
    """
    timeout = kwargs.pop("timeout", REQUEST_TIMEOUT_SECONDS)
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.Timeout:
        st.error("The request timed out. Please try again.")
        return None
    except requests.ConnectionError:
        st.error(f"Could not reach the backend at {BACKEND_URL}. Is it running?")
        return None
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None

    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text) if resp.content else resp.text
        st.error(f"Request failed ({resp.status_code}): {detail}")
        return None
    return resp


def _max_papers_control(key_prefix: str) -> int:
    """A slider and a number input kept in sync via shared session_state keys."""
    slider_key = f"{key_prefix}_max_papers_slider"
    input_key = f"{key_prefix}_max_papers_input"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 15
    if input_key not in st.session_state:
        st.session_state[input_key] = 15

    def _sync_from_slider() -> None:
        st.session_state[input_key] = st.session_state[slider_key]

    def _sync_from_input() -> None:
        st.session_state[slider_key] = st.session_state[input_key]

    st.markdown("**Max papers**")
    slider_col, number_col = st.columns([3, 1])
    with slider_col:
        st.slider(
            "Max papers",
            *MAX_PAPERS_BOUNDS,
            key=slider_key,
            on_change=_sync_from_slider,
            label_visibility="collapsed",
        )
    with number_col:
        st.number_input(
            "Max papers (exact)",
            *MAX_PAPERS_BOUNDS,
            key=input_key,
            on_change=_sync_from_input,
            label_visibility="collapsed",
        )
    return int(st.session_state[slider_key])


def _search_filters_controls(key_prefix: str) -> dict[str, Any]:
    year_min = st.number_input(
        "Year from", min_value=1900, max_value=2100, value=2018, key=f"{key_prefix}_year_min"
    )
    year_max = st.number_input(
        "Year to", min_value=1900, max_value=2100, value=2025, key=f"{key_prefix}_year_max"
    )
    article_type = st.multiselect(
        "Article types", ARTICLE_TYPE_OPTIONS, key=f"{key_prefix}_article_type"
    )
    max_papers = _max_papers_control(key_prefix)
    return {
        "year_min": year_min,
        "year_max": year_max,
        "article_types": article_type,
        "max_papers": max_papers,
    }


def _render_evidence_summary_tab(tab: Any, report_markdown: str, machine_json: dict) -> None:
    with tab:
        stats = evidence_summary_stats(machine_json)
        if stats["total_studies"]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Studies synthesized", stats["total_studies"])
            c2.metric("Strongest evidence", stats["strongest_level_label"].split("—")[0].strip())
            c3.metric("References", stats["total_references"])
            c4.metric("Year range", stats["year_range"])

            if stats["level_counts"]:
                with st.expander("Evidence level breakdown"):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Evidence level": list(stats["level_counts"].keys()),
                                "Studies": list(stats["level_counts"].values()),
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            st.divider()

        if report_markdown:
            st.markdown(report_markdown)
        else:
            st.info("No report was generated for this query.")


def _render_study_comparison_tab(tab: Any, machine_json: dict) -> None:
    with tab:
        narrative = comparison_narrative(machine_json)
        if any(narrative.values()):
            for label, key in (
                ("✅ Agreements", "agreements"),
                ("⚠️ Disagreements", "disagreements"),
                ("📈 Emerging trends", "trends"),
            ):
                if narrative[key]:
                    st.markdown(f"**{label}**")
                    for item in narrative[key]:
                        st.markdown(f"- {item}")

        matrix = comparison_matrix_rows(machine_json)
        if matrix:
            st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)
        else:
            st.info("No cross-study comparison is available for this query.")


def _render_study_details_tab(tab: Any, machine_json: dict) -> None:
    with tab:
        rows = study_detail_rows(machine_json)
        if not rows:
            st.info("No studies were retrieved for this query.")
            return
        st.caption(
            "Full text isn't fetched here (publisher access restrictions) — each study "
            "links out to its source so you can read the original."
        )
        for row in rows:
            title = row["title"] or f"PMID {row['pmid']}"
            with st.expander(f"{title}  ·  PMID {row['pmid']}"):
                st.markdown(
                    f"**Journal:** {row['journal'] or '—'}  ·  "
                    f"**Year:** {row['publication_year'] or '—'}  ·  "
                    f"**Evidence level:** {row['evidence_level']}"
                )
                link_bits = []
                if row["pmid_url"]:
                    link_bits.append(f"[📖 Read on PubMed]({row['pmid_url']})")
                if row["doi_url"]:
                    link_bits.append(f"[🔗 Publisher / DOI page]({row['doi_url']})")
                if link_bits:
                    st.markdown("  ·  ".join(link_bits))
                if row["abstract"]:
                    st.markdown(f"**Abstract:** {row['abstract']}")
                for label, key in (
                    ("Objective", "objective"),
                    ("Population", "population"),
                    ("Intervention", "intervention"),
                    ("Comparator", "comparator"),
                    ("Main findings", "main_findings"),
                    ("Statistical significance", "statistical_significance"),
                    ("Limitations", "limitations"),
                ):
                    if row[key]:
                        st.markdown(f"**{label}:** {row[key]}")
                if row["outcomes"]:
                    st.markdown(f"**Outcomes:** {', '.join(row['outcomes'])}")
                if row["strength"] or row["bias_risk"]:
                    st.markdown(
                        f"**Strength of evidence:** {row['strength'] or '—'}  ·  "
                        f"**Risk of bias:** {row['bias_risk'] or '—'}"
                    )
                if row["confidence_reasoning"]:
                    st.caption(row["confidence_reasoning"])


def _render_references_tab(tab: Any, machine_json: dict) -> None:
    with tab:
        rows = reference_rows(machine_json)
        if not rows:
            st.info("No references are available for this query.")
            return
        for i, row in enumerate(rows, start=1):
            links = []
            if row["pmid_url"]:
                links.append(f"[PubMed]({row['pmid_url']})")
            if row["doi_url"]:
                links.append(f"[DOI]({row['doi_url']})")
            link_str = "  ·  ".join(links)
            st.markdown(f"{i}. {row['vancouver']}" + (f"  ·  {link_str}" if link_str else ""))


def _render_research_result(tabs: list[Any], data: dict) -> None:
    warnings = data.get("warnings") or []
    if warnings:
        st.warning(
            "This is a **partial result** — one or more pipeline steps reported "
            "issues:\n\n" + "\n".join(f"- {w}" for w in warnings)
        )

    machine_json = data.get("machine_json") or {}
    if not machine_json.get("studies"):
        st.info("No studies were retrieved for this question. Try broadening your filters.")

    _render_evidence_summary_tab(tabs[0], data.get("report_markdown", ""), machine_json)
    _render_study_comparison_tab(tabs[1], machine_json)
    _render_study_details_tab(tabs[2], machine_json)
    _render_references_tab(tabs[3], machine_json)


RESULT_TAB_LABELS = [
    "📋 Evidence Summary",
    "⚖️ Study Comparison",
    "📑 Study Details",
    "🔗 References",
]


# --- Normal Search mode (unchanged) -------------------------------------


def _render_normal_search_mode() -> None:
    with st.sidebar:
        st.header("🔎 Search filters")
        filters = _search_filters_controls(key_prefix="normal")

    question_col, keywords_col = st.columns([3, 2])
    with question_col:
        question = st.text_input(
            "Clinical question", placeholder="What are recent treatments for keratoconus?"
        )
    with keywords_col:
        keywords_raw = st.text_input(
            "Additional keywords (optional)",
            placeholder="e.g. crosslinking, pediatric, RCT",
            help="Comma-separated terms the search should be biased toward, "
            "on top of the question.",
        )
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    run = st.button("Synthesize evidence", type="primary", use_container_width=False)

    tabs = st.tabs(RESULT_TAB_LABELS)

    if run and question:
        with st.spinner("Running multi-agent evidence synthesis…"):
            resp = _api_request(
                "POST",
                "/research",
                json={"question": question, "filters": {**filters, "keywords": keywords}},
            )
        if resp is not None:
            _render_research_result(tabs, resp.json())
    else:
        with tabs[0]:
            st.caption("Enter a clinical question and press *Synthesize evidence*.")


# --- Projects mode -------------------------------------------------------


def _render_project_search_tab(project_id: str) -> None:
    filters = _search_filters_controls(key_prefix=f"proj_{project_id}")

    question_col, keywords_col = st.columns([3, 2])
    with question_col:
        question = st.text_input(
            "Clinical question",
            placeholder="What are recent treatments for keratoconus?",
            key=f"proj_{project_id}_question",
        )
    with keywords_col:
        keywords_raw = st.text_input(
            "Additional keywords (optional)",
            placeholder="e.g. crosslinking, pediatric, RCT",
            key=f"proj_{project_id}_keywords",
        )
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

    result_key = f"proj_{project_id}_last_result"
    run = st.button("Synthesize evidence", type="primary", key=f"proj_{project_id}_run")
    if run and question:
        with st.spinner("Running multi-agent evidence synthesis…"):
            resp = _api_request(
                "POST",
                f"/projects/{project_id}/research",
                json={"question": question, "filters": {**filters, "keywords": keywords}},
            )
        if resp is not None:
            # Other tabs (Paper History) fetched the project's detail before this
            # run's newly-embedded papers committed — rerun so they're immediately
            # visible rather than only catching up on the next unrelated interaction.
            st.session_state[result_key] = resp.json()
            st.rerun()

    if result_key in st.session_state:
        result_tabs = st.tabs(RESULT_TAB_LABELS)
        _render_research_result(result_tabs, st.session_state[result_key])
    else:
        st.caption("Enter a clinical question and press *Synthesize evidence*.")


def _render_project_history_tab(detail: dict) -> None:
    history_rows = project_history_rows(detail.get("history") or [])
    document_rows = project_document_rows(detail.get("documents") or [])

    st.markdown(f"**{len(document_rows)} paper(s)** in this project's corpus (used for chat).")
    if document_rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Title": d["title"] or "Untitled",
                        "Source": d["source"],
                        "PMID": d["pmid"] or "—",
                    }
                    for d in document_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No papers yet — search PubMed or upload a PDF to get started.")

    st.divider()
    st.markdown("**Search history**")
    if not history_rows:
        st.caption("No searches run in this project yet.")
    for row in history_rows:
        label = f"{row['question']}  ·  {row['study_count']} paper(s)  ·  {row['created_at']}"
        with st.expander(label):
            if not row["studies"]:
                st.caption("No studies retrieved for this search.")
            for study in row["studies"]:
                title = study.get("title") or "Untitled"
                pmid = study.get("pmid", "—")
                st.markdown(f"- **{title}** · PMID {pmid}")


def _render_project_upload_tab(project_id: str) -> None:
    st.caption(
        "Upload a paper PDF to add it to this project's chat corpus — useful for "
        "papers not on PubMed, or full-text PDFs you already have."
    )

    flash_key = f"proj_{project_id}_upload_success"
    if flash_key in st.session_state:
        st.success(f"Added **{st.session_state.pop(flash_key)}** to this project.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key=f"proj_{project_id}_uploader")
    if uploaded is not None and st.button("Add to project", key=f"proj_{project_id}_upload_btn"):
        with st.spinner("Extracting and indexing…"):
            resp = _api_request(
                "POST",
                f"/projects/{project_id}/documents",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
        if resp is not None:
            # Other tabs (Paper History) fetched the project's detail before this
            # run's upload committed — rerun so they reflect it immediately rather
            # than only catching up on the user's next unrelated interaction.
            st.session_state[flash_key] = resp.json()["title"]
            st.rerun()


def _render_project_chat_tab(project_id: str) -> None:
    history_resp = _api_request("GET", f"/projects/{project_id}/chat")
    history = history_resp.json() if history_resp is not None else []

    if not history:
        st.caption(
            "Ask a question grounded only in this project's papers — PubMed abstracts/"
            "findings and any uploaded PDFs. Like NotebookLM, answers cite only what's "
            "actually in this project's corpus, not general knowledge."
        )

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about this project's papers…", key=f"proj_{project_id}_chat_in")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = _api_request(
                    "POST", f"/projects/{project_id}/chat", json={"message": prompt}
                )
            if resp is not None:
                st.markdown(resp.json()["reply"])
            else:
                st.markdown("_Failed to get a response — see error above._")


def _render_projects_mode() -> None:
    with st.sidebar:
        st.header("📁 Projects")
        projects_resp = _api_request("GET", "/projects")
        projects = projects_resp.json() if projects_resp is not None else []

        selected_id = st.session_state.get("selected_project_id")
        ids = [p["id"] for p in projects]
        if selected_id not in ids:
            selected_id = ids[0] if ids else None
            st.session_state["selected_project_id"] = selected_id

        if projects:
            labels = {p["id"]: p["name"] for p in projects}
            index = ids.index(selected_id) if selected_id in ids else 0
            selected_id = st.selectbox(
                "Select project", ids, index=index, format_func=lambda pid: labels.get(pid, pid)
            )
            st.session_state["selected_project_id"] = selected_id

            delete_clicked = st.button("🗑️ Delete this project")
            if delete_clicked and _api_request("DELETE", f"/projects/{selected_id}") is not None:
                st.session_state["selected_project_id"] = None
                st.rerun()
        else:
            st.caption("No projects yet — create one below.")

        with st.expander("➕ New project", expanded=not projects):
            new_name = st.text_input("Project name", key="new_project_name_input")
            if st.button("Create project", key="create_project_btn") and new_name.strip():
                create_resp = _api_request("POST", "/projects", json={"name": new_name.strip()})
                if create_resp is not None:
                    st.session_state["selected_project_id"] = create_resp.json()["id"]
                    st.rerun()

    selected_id = st.session_state.get("selected_project_id")
    if not selected_id:
        st.info("Create or select a project from the sidebar to get started.")
        return

    detail_resp = _api_request("GET", f"/projects/{selected_id}")
    if detail_resp is None:
        return
    detail = detail_resp.json()

    st.subheader(f"📁 {detail['project']['name']}")
    project_tabs = st.tabs(["🔎 Search", "📚 Paper History", "📤 Upload PDF", "💬 Chat"])
    with project_tabs[0]:
        _render_project_search_tab(selected_id)
    with project_tabs[1]:
        _render_project_history_tab(detail)
    with project_tabs[2]:
        _render_project_upload_tab(selected_id)
    with project_tabs[3]:
        _render_project_chat_tab(selected_id)


# --- Mode switch ---------------------------------------------------------

mode = st.sidebar.radio("Mode", ["🔍 Normal Search", "📁 Projects"], label_visibility="collapsed")
st.sidebar.divider()

if mode == "🔍 Normal Search":
    _render_normal_search_mode()
else:
    _render_projects_mode()
