"""Feature: understand the question.
Decides what the student wants (Chat / Explain / Summarize / Quiz)
and write a search query, using the chat history for follow-ups."""
from pydantic import BaseModel, Field

from src.agent import config
from src.agent.state import MessageIntent, StudyState


class IntentClassifier(BaseModel):
    message_intent: MessageIntent
    search_query: str = Field(
        description="A short, complete search query for the textbook (a few keywords). Empty for Chat.")


CLASSIFIER_PROMPT = """You are the router for a study assistant. Classify the user's latest message into exactly one category:
- Explain: asks to explain, describe, or answer a question about a concept or topic
  (for example "what is X", "how does X work", "explain the steps of X").
- Summarize: asks to summarize or shorten a topic or content.
- Quiz: asks for a quiz, practice questions, or to test their knowledge.
- Plan: asks for a study plan, a study schedule, or how to organize their studying.
- Chat: ONLY greetings, thanks, small talk, or requests clearly unrelated to studying (like recipes).
The message may contain spelling mistakes; focus on what the user means.
If you are unsure between Chat and another category, choose the other category.
If the message refers to something earlier ("that", "it") but there are no earlier messages, choose Chat.

Also write search_query: a short, complete search query for the textbook (a few keywords).
- If the latest message refers to earlier messages ("that", "it", "explain more"),
  use the recent conversation to find the topic.
- Fix spelling mistakes.
- For Chat, leave search_query empty."""


def classify_intent(state: StudyState):
    question = state["messages"][-1].content

   
    recent = state["messages"][-7:-1] 
    history_text = "\n".join(f"{m.type}: {m.content}" for m in recent) or "(no earlier messages)"

    structured_llm = config.llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": f"Recent conversation:\n{history_text}\n\nLatest message: {question}"},
    ])

  
    return {
        "message_intent": result.message_intent,
        "search_query": result.search_query.strip() or question,
        "retrieved_docs": [],
        "draft_answer": "",
        "missing": "",
        "retry_count": 0,
        "quiz": None,
        "problems": [],
        "failed_checks": 0,
        "plan": None,         
        "approval": None,     
        "action_result": "", 
        "review_issues": [], 
        "review_count": 0,
    }