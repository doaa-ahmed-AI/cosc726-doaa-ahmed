"""Decisions: which node runs next. Every arrow that has a choice is decided here."""
from src.agent import config
from src.agent.state import StudyState

AGENT_FOR_INTENT = {
    "Explain": "explanation agent",
    "Summarize": "summary agent",
    "Quiz": "quiz agent",
    "Plan": "plan agent",  # PHASE 3 - STEP 2
}


def route_after_classifier(state: StudyState) -> str:
    """Small talk goes to the chat agent; everything else searches the book first."""
    if state["message_intent"] == "Chat":
        return "chat agent"
    return "retrieve"


def route_after_retrieve(state: StudyState) -> str:
    """Pick the agent that matches what the student asked for."""
    return AGENT_FOR_INTENT[state["message_intent"]]


def route_after_validate(state: StudyState) -> str:
    
    if state["problems"] and state["failed_checks"] <= config.MAX_FIXES:
        return AGENT_FOR_INTENT[state["message_intent"]]
    
    if state["missing"] and state["retry_count"] < config.MAX_RETRIES:
        return "rewrite query"
   
    if state["message_intent"] == "Plan" and state.get("plan") and state["plan"]["sessions"]:
        return "ask approval"
   
    if state["draft_answer"].strip() != config.NOT_COVERED:
        return "critic"
    
    return "finalize"



def route_after_critic(state: StudyState) -> str:
    """The critic rejected the answer -> back to the same agent (limited times). Otherwise finish."""
    if state["review_issues"] and state["review_count"] <= config.MAX_REVIEWS:
        return AGENT_FOR_INTENT[state["message_intent"]]
    return "finalize"



def route_after_approval(state: StudyState) -> str:
    """Only an approved (or edited) plan may be shared."""
    if state["approval"] in ("approve", "edit"):
        return "share plan"
    return "finalize"