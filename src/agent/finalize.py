"""Feature: put the final answer into the chat (only the last draft, never the half-answers)."""
from src.agent.state import StudyState


def finalize(state: StudyState):
    answer = state["draft_answer"]
    # if the answer still fails the checks, say so honestly
    if state.get("problems"):
        answer += ("\n\n⚠️ This answer did not pass every automatic check "
                   "(citations or quiz format), so double-check it against the book.")
    return {"messages": [{"role": "assistant", "content": answer}]}