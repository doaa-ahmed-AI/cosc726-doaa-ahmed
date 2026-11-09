"""Feature: check answers before the student see them.
- citation check: every cited page must be a page the search really found
- quiz check: every question has 4 options and an explanation"""
import re

from src.agent import config
from src.agent.state import StudyState


PAGE_PATTERN = re.compile(r"\bpages?\s+(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE)


def find_cited_pages(text: str) -> set[int]:
    pages = set()
    for start, end in PAGE_PATTERN.findall(text):
        pages.add(int(start))
        if end:
            pages.add(int(end))
    return pages


def find_problems(answer: str, docs: list[dict], quiz: dict | None) -> list[str]:
    if answer.strip() == config.NOT_COVERED:
        return []  # an honest "not covered" needs no citations

    problems = []
    found_pages = {d["page"] for d in docs}
    cited_pages = find_cited_pages(answer)

    if not cited_pages:
        problems.append("The answer has no citations. Cite a page after every point.")
    for page in sorted(cited_pages - found_pages):
        problems.append(
            f"The answer cites page {page}, but that page is not in the course material you were given.")

    if quiz is not None:
        for i, q in enumerate(quiz["questions"], start=1):
            if len(q["options"]) != 4:
                problems.append(f"Question {i} has {len(q['options'])} options; it must have exactly 4.")
            if not q["explanation"].strip():
                problems.append(f"Question {i} has no explanation.")

    return problems


def feedback_text(state: StudyState) -> str:
    """If the last answer failed the checks (or the critic rejected it), tell the LLM what to fix."""
    issues = list(state.get("problems") or [])
    issues += [f"Reviewer: {i}" for i in (state.get("review_issues") or [])]  # PHASE 3 - STEP 3
    if not issues:
        return ""
    lines = "\n".join(f"- {p}" for p in issues)
    return f"\n\nYour previous answer had these problems. Fix them:\n{lines}"


def validate(state: StudyState):
    problems = find_problems(state["draft_answer"], state["retrieved_docs"], state.get("quiz"))
    failed_checks = state["failed_checks"] + (1 if problems else 0)
    return {"problems": problems, "failed_checks": failed_checks}