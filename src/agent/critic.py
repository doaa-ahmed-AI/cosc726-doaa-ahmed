"""Feature: the critic.
A second LLM role with a different goal: it does not write answers, it CHECKS them.
It reads the answer next to the pages it cites, and rejects claims those pages do not support."""
from pydantic import BaseModel, Field

from src.agent import config
from src.agent.retrieval import format_docs
from src.agent.state import StudyState
from src.agent.validation import find_cited_pages


class Review(BaseModel):
    supported: bool = Field(description="True if every claim in the answer is supported by the cited pages.")
    issues: list[str] = Field(
        description="Each unsupported claim: what it says, the page it cites, and what that page really says. "
                    "Empty if everything is supported.")


CRITIC_PROMPT = """You are a strict reviewer for a study assistant. You do not write answers; you check them.

Course material (only the pages the answer cites):
{material}

Answer to check:
{answer}

For each claim in the answer, check that the page it cites really says it.
- Use ONLY the course material above, never your own knowledge.
- Ignore style, wording and length; only check facts and citations.
- A claim is unsupported if the cited page does not say it, or says something different.
- If every claim is supported, set "supported" to true and leave "issues" empty.
- Otherwise set "supported" to false and list each unsupported claim, the page it cites,
  and what that page really says."""


def review_answer(answer: str, docs: list[dict]) -> Review:
    """Check one answer against the pieces of the pages it cites."""
    cited = find_cited_pages(answer)
    evidence = [d for d in docs if d["page"] in cited] or docs  # only the pages the answer cites
    prompt = CRITIC_PROMPT.format(material=format_docs(evidence), answer=answer)
    return config.llm.with_structured_output(Review).invoke([{"role": "user", "content": prompt}])


def critic(state: StudyState):
    review = review_answer(state["draft_answer"], state["retrieved_docs"])
    issues = [] if review.supported else (
        review.issues or ["The reviewer found a claim the cited pages do not support."])
    return {
        "review_issues": issues,
        "review_count": state["review_count"] + (1 if issues else 0),
    }