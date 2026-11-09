from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import AnyMessage, add_messages

MessageIntent = Literal["Chat", "Explain", "Summarize", "Quiz", "Plan"]


class StudyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: MessageIntent | None
    search_query: str
    retrieved_docs: list[dict]
    draft_answer: str
    missing: str
    retry_count: int
    quiz: dict | None
    problems: list[str]
    failed_checks: int
    plan: dict | None           
    approval: str | None      
    action_result: str
    review_issues: list[str]   
    review_count: int
