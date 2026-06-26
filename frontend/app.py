"""Streamlit frontend for the Medical Research Agent.

Phase 1 establishes the layout (filters, five output tabs, disclaimer, loading
indicators). It talks to the backend's /research endpoint, which is wired in
later phases.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DISCLAIMER = (
    "For research and educational purposes only. Clinical decisions should rely "
    "on professional judgment and full-text evidence review."
)

st.set_page_config(page_title="Medical Research Agent", page_icon="🔬", layout="wide")
st.title("🔬 Medical Research Agent")
st.warning(DISCLAIMER)

with st.sidebar:
    st.header("Search filters")
    year_min = st.number_input("Year from", min_value=1900, max_value=2100, value=2018)
    year_max = st.number_input("Year to", min_value=1900, max_value=2100, value=2025)
    article_type = st.multiselect(
        "Article types",
        ["Randomized Controlled Trial", "Meta-Analysis", "Systematic Review", "Review"],
    )
    max_papers = st.slider("Max papers", 1, 50, 15)

question = st.text_input("Clinical question", placeholder="What are recent treatments for keratoconus?")
run = st.button("Synthesize evidence", type="primary")

tabs = st.tabs(
    ["Evidence Summary", "Study Comparison", "Study Details", "References", "Raw JSON"]
)

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
                    },
                },
                timeout=300,
            )
            if resp.status_code == 501:
                st.info("Backend pipeline is not yet implemented (Phases 3-4).")
            else:
                resp.raise_for_status()
                data = resp.json()
                with tabs[0]:
                    st.markdown(data.get("report_markdown", ""))
                with tabs[4]:
                    st.json(data.get("machine_json", {}))
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
else:
    with tabs[0]:
        st.caption("Enter a clinical question and press *Synthesize evidence*.")
