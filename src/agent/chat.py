"""Feature: greetings and small talk (no course material needed)."""
from src.agent import config
from src.agent.state import StudyState

CHAT_PROMPT = ("You are a friendly study assistant. Reply to greetings and small talk briefly. "
               "If the user asks for something unrelated to studying (like recipes), "
               "politely say you are here to help with their course material.")


def prompt_llm_chat(state: StudyState):
    messages = [{"role": "system", "content": CHAT_PROMPT}] + state["messages"]
    response = config.llm.invoke(messages)
    return {"messages": [response]}