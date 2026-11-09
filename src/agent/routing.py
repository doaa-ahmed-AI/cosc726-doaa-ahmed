"""Decisions: which node runs next. Every arrow that has a choice is decided here."""
from src.agent import config
from src.agent.state import StudyState

AGENT_FOR_INTENT = {
    "Explain": "explanation agent",
    "Summarize": "summary agent",
    "Quiz": "quiz agent",
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
   
    return "finalize"