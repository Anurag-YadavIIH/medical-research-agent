# Evaluation Module

Phase 7 deliverable. Measures:

- **Citation completeness** — every claim in the summary maps to a retrieved study.
- **Extraction accuracy** — extracted fields vs. a labelled gold set.
- **Evidence consistency** — agreement between assigned levels and study design.
- **Hallucination checks** — references and PMIDs must exist in the retrieved set.

Datasets live in `datasets/`. Reports are emitted to `reports/` as JSON + Markdown.
