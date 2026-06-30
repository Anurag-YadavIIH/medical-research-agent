"""Streamlit frontend for the Medical Research Agent.

This file only calls the backend's /research endpoint and renders the response
— all synthesis, citation handling, and evidence grading happen server-side.
The frontend container doesn't install the backend package (see
docker/frontend.Dockerfile), so BACKEND_URL is read from the environment
directly rather than via config.Settings.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from transforms import (
    comparison_matrix_rows,
    comparison_narrative,
    evidence_summary_stats,
    reference_rows,
    study_detail_rows,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DISCLAIMER = (
    "For research and educational purposes only. Clinical decisions should rely "
    "on professional judgment and full-text evidence review."
)
REQUEST_TIMEOUT_SECONDS = 300
MAX_PAPERS_BOUNDS = (1, 50)

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

with st.sidebar:
    st.header("🔎 Search filters")
    year_min = st.number_input("Year from", min_value=1900, max_value=2100, value=2018)
    year_max = st.number_input("Year to", min_value=1900, max_value=2100, value=2025)
    article_type = st.multiselect(
        "Article types",
        ["Randomized Controlled Trial", "Meta-Analysis", "Systematic Review", "Review"],
    )

    st.markdown("**Max papers**")
    if "max_papers_slider" not in st.session_state:
        st.session_state["max_papers_slider"] = 15
    if "max_papers_input" not in st.session_state:
        st.session_state["max_papers_input"] = 15

    def _sync_from_slider() -> None:
        st.session_state["max_papers_input"] = st.session_state["max_papers_slider"]

    def _sync_from_input() -> None:
        st.session_state["max_papers_slider"] = st.session_state["max_papers_input"]

    slider_col, number_col = st.columns([3, 1])
    with slider_col:
        st.slider(
            "Max papers",
            *MAX_PAPERS_BOUNDS,
            key="max_papers_slider",
            on_change=_sync_from_slider,
            label_visibility="collapsed",
        )
    with number_col:
        st.number_input(
            "Max papers (exact)",
            *MAX_PAPERS_BOUNDS,
            key="max_papers_input",
            on_change=_sync_from_input,
            label_visibility="collapsed",
        )
    max_papers = st.session_state["max_papers_slider"]

question_col, keywords_col = st.columns([3, 2])
with question_col:
    question = st.text_input(
        "Clinical question", placeholder="What are recent treatments for keratoconus?"
    )
with keywords_col:
    keywords_raw = st.text_input(
        "Additional keywords (optional)",
        placeholder="e.g. crosslinking, pediatric, RCT",
        help="Comma-separated terms the search should be biased toward, on top of the question.",
    )
keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]

run = st.button("Synthesize evidence", type="primary", use_container_width=False)

tabs = st.tabs(["📋 Evidence Summary", "⚖️ Study Comparison", "📑 Study Details", "🔗 References"])


def _render_evidence_summary_tab(report_markdown: str, machine_json: dict) -> None:
    with tabs[0]:
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


def _render_study_comparison_tab(machine_json: dict) -> None:
    with tabs[1]:
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


def _render_study_details_tab(machine_json: dict) -> None:
    with tabs[2]:
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


def _render_references_tab(machine_json: dict) -> None:
    with tabs[3]:
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


def _render_result(data: dict) -> None:
    warnings = data.get("warnings") or []
    if warnings:
        st.warning(
            "This is a **partial result** — one or more pipeline steps reported "
            "issues:\n\n" + "\n".join(f"- {w}" for w in warnings)
        )

    machine_json = data.get("machine_json") or {}
    if not machine_json.get("studies"):
        st.info("No studies were retrieved for this question. Try broadening your filters.")

    _render_evidence_summary_tab(data.get("report_markdown", ""), machine_json)
    _render_study_comparison_tab(machine_json)
    _render_study_details_tab(machine_json)
    _render_references_tab(machine_json)


if run and question:
    with st.spinner("Running multi-agent evidence synthesis…"):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/research",
                json={
                    "question": question,
                    "filters": {
                        "year_min": year_min,
                        "year_max": year_max,
                        "article_types": article_type,
                        "max_papers": max_papers,
                        "keywords": keywords,
                    },
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout:
            with tabs[0]:
                st.error(
                    "The request timed out. Evidence synthesis can take a while for "
                    "broad questions — try narrowing your filters or asking again."
                )
        except requests.ConnectionError:
            with tabs[0]:
                st.error(f"Could not reach the backend at {BACKEND_URL}. Is it running?")
        except requests.RequestException as exc:
            with tabs[0]:
                st.error(f"Request failed: {exc}")
        else:
            if resp.status_code == 503:
                with tabs[0]:
                    st.error(resp.json().get("detail", "The backend is not configured."))
            elif resp.status_code >= 400:
                with tabs[0]:
                    detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                    st.error(f"Request failed ({resp.status_code}): {detail}")
            else:
                _render_result(resp.json())
else:
    with tabs[0]:
        st.caption("Enter a clinical question and press *Synthesize evidence*.")
