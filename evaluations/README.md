# Evaluation Module

Phase 7 deliverable. Four metrics, computed in `metrics.py` as pure functions
over a run's `machine_json`:

- **citation_completeness** — every PMID/DOI cited in the narrative or listed
  as a reference must exist in the retrieved set. A cited-but-absent ID is a
  hard failure, listed by name — scoring 1.0 while citing a fabricated ID is
  structurally impossible.
- **extraction_accuracy** — extracted fields vs. a labelled gold set.
  `study_design`/`sample_size` are matched exactly; free-text fields
  (population, intervention, comparator, main_findings,
  statistical_significance) are matched fuzzily (`difflib.SequenceMatcher`
  ratio ≥ 0.6). A field gold doesn't label is skipped, not penalized.
- **evidence_consistency** — reuses the real `deterministic_level()` from
  `agents/evidence_evaluator.py` (not a re-implementation) to check the
  assigned evidence level against publication_types. A mismatch is a hard
  failure; a publication type the map can't classify (UNGRADED) is a
  surfaced gap, not a silent pass.
- **hallucination_check** — every PMID/DOI anywhere in the output (references,
  comparison's PMID lists, narrative citations) must be in the retrieved set,
  and every sample-size/p-value-shaped numeric claim in the narrative must
  trace to an extracted field.

## Running it

```bash
make eval                    # fixtures only — no network, no LLM key needed
make eval ARGS="--case keratoconus_crosslinking"
make eval ARGS="--live"      # real NCBI/CrossRef + a real configured LLM key
```

Reports land in `reports/<timestamp>.{json,md}` (gitignored). Exit code is 1
only for *critical* failures (a fabricated citation, or an evidence-level
mismatch) — extraction misses and untraceable-claim flags are real findings
worth reading but don't fail the run.

## Fixtures

`datasets/*.json` carry labelled gold extractions plus a deterministic LLM
stub's answers. `fixtures/<case_id>/` hold **real, recorded** NCBI
esearch/efetch and CrossRef responses for actual published trials — not
fabricated data. `runner.py` runs the real `build_research_graph()` against
these fixtures with PubMed/CrossRef mocked via respx and every agent's LLM
call replaced by the per-case deterministic stub. `run_case_live()` makes no
mocking changes at all and is what `--live` calls — that's the true
end-to-end path once a real LLM key is supplied.
