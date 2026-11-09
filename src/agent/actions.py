"""Feature: actions the agent can take, with permission tiers.
- read-only:      runs automatically, cannot change anything
- consequential:  affects other people or cannot be undone -> needs a human's approval first"""
import json
import os
from datetime import datetime

from src.agent import config
from src.agent.state import StudyState


TOOLS = {
    "search_course_material": {"node": "retrieve", "tier": "read-only", "needs_approval": False},
    "share_study_plan": {"node": "share plan", "tier": "consequential", "needs_approval": True},
}


def share_study_plan(state: StudyState):
    """MOCK action: 'sends' the plan to the study group by saving it to a file. Nothing is really sent."""
    
    if TOOLS["share_study_plan"]["needs_approval"] and state.get("approval") not in ("approve", "edit"):
        raise PermissionError("share_study_plan needs a human's approval first")

    os.makedirs(os.path.dirname(config.OUTBOX_FILE), exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "to": config.STUDY_GROUP,
        "approval": state["approval"],
        "plan": state["draft_answer"],
    }
    with open(config.OUTBOX_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"action_result": f"✅ Shared with your study group ({config.STUDY_GROUP}). "
                             f"(Mock action: saved to {config.OUTBOX_FILE}; nothing was really sent.)"}