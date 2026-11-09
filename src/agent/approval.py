"""Feature: the human approval gate .
The graph PAUSES here and waits for the student to approve, edit or reject,
before the consequential action (sharing the plan) is allowed to run."""
from langgraph.types import interrupt

from src.agent import config
from src.agent.state import StudyState


def ask_approval(state: StudyState):
   
    decision = interrupt({
        "action": "share_study_plan",
        "description": f"Share this study plan with your study group ({config.STUDY_GROUP})?",
        "plan": state["draft_answer"],
    })

    
    choice = decision.get("choice")
    if choice == "approve":
        return {"approval": "approve"}
    if choice == "edit":
        return {"approval": "edit", "draft_answer": decision["plan"]} 

    return {"approval": "reject",
            "action_result": "Not shared: you rejected it, so the plan stays private."}
