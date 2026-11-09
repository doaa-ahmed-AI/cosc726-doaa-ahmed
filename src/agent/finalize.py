"""Feature: put the final answer into the chat (only the last draft, never the half-answers)."""
from src.agent.state import StudyState


def finalize(state: StudyState):
    answer = state["draft_answer"]
  
    if state.get("problems"):
        answer += ("\n\n⚠️ This answer did not pass every automatic check "
                   "(citations or quiz format), so double-check it against the book.")

    if state.get("action_result"):
        answer += "\n\n" + state["action_result"]   
    return {"messages": [{"role": "assistant", "content": answer}]}