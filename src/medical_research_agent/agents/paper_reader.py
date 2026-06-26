"""Extract structured clinical content from each study abstract."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from medical_research_agent.agents.base import BaseAgent
from medical_research_agent.llm.factory import get_chat_model
from medical_research_agent.models.state import ResearchState
from medical_research_agent.models.study import ExtractedStudy

_SYSTEM_PROMPT = (
    "You are extracting structured clinical information strictly from the abstract "
    "text provided. Use ONLY information explicitly stated in the abstract. If a "
    "field is not stated, leave it empty — never infer, estimate, or guess values "
    "(including sample sizes or findings) that are not present in the text.\n\n"
    "For sample_size: put a number ONLY if the abstract reports a single participant/"
    "patient headcount (e.g. 50 patients, 140 eyes). If it instead reports a count of "
    "studies, trials, or any other non-participant quantity (e.g. '36 studies', "
    "'12 RCTs'), or several different counts with no single headcount, leave "
    "sample_size null and put the verbatim text in sample_size_description instead."
)


class PaperReaderAgent(BaseAgent):
    """Extract structured clinical content from each study abstract."""

    name = "paper_reader"

    async def run(self, state: ResearchState) -> dict[str, object]:
        model = get_chat_model().with_structured_output(ExtractedStudy)
        extracted: list[ExtractedStudy] = []
        errors: list[str] = []

        for study in state.studies:
            if not study.abstract.strip():
                extracted.append(ExtractedStudy(pmid=study.pmid))
                continue
            try:
                result = cast(
                    ExtractedStudy,
                    await model.ainvoke(
                        [
                            SystemMessage(content=_SYSTEM_PROMPT),
                            HumanMessage(
                                content=f"Abstract (PMID {study.pmid}):\n{study.abstract}"
                            ),
                        ]
                    ),
                )
                extracted.append(result.model_copy(update={"pmid": study.pmid}))
            except Exception as exc:  # noqa: BLE001 - per-study failure shouldn't drop the rest
                errors.append(f"{self.name}: failed to extract PMID {study.pmid}: {exc}")

        update: dict[str, object] = {"extracted": extracted}
        if errors:
            update["errors"] = errors
        return update
