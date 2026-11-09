"""Feature: make a study plan from the course material."""
from pydantic import BaseModel, Field

from src.agent import config
from src.agent.retrieval import format_docs
from src.agent.state import StudyState
from src.agent.validation import feedback_text


class StudySession(BaseModel):
    day: str = Field(description='When to study, e.g. "Day 1".')
    topic: str = Field(description="A topic from the course material.")
    page: int = Field(description="The page (from the material headers) where this topic starts.")
    activity: str = Field(description="What to do, e.g. read and summarize, or practice with a quiz.")


class StudyPlan(BaseModel):
    sessions: list[StudySession]
    missing: str = Field(description="What the material did not cover. Empty string if nothing is missing.")


def format_plan(plan: StudyPlan) -> str:
    """Turn the plan data into readable text."""
    if not plan.sessions:
        return config.NOT_COVERED
    lines = ["**Your study plan**\n"]
    for s in plan.sessions:
        lines.append(f"- **{s.day}: {s.topic}** (page {s.page}): {s.activity}")
    return "\n".join(lines)


def make_plan(state: StudyState):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    prompt = f"""Course material:
{context}

Request: {question}

Rules:
- Make a study plan using ONLY topics from the course material above.
- Each session has a day (e.g. "Day 1"), a topic, the page where that topic starts
  (from the material headers), and an activity (e.g. read and summarize, practice with a quiz).
- If the request says how many days or sessions, follow it; otherwise make 5 sessions.
- Put earlier topics first.
- If the material covers only part of the request, plan that part and put the rest in "missing".
- If the material is about a completely different topic, return no sessions and write the topic in "missing".{feedback_text(state)}"""

    messages = ([{"role": "system", "content": "You are a helpful assistant that makes study plans from course material."}]
                + history
                + [{"role": "user", "content": prompt}])

    structured_llm = config.llm.with_structured_output(StudyPlan)
    plan = structured_llm.invoke(messages)

    return {
        "draft_answer": format_plan(plan),
        "missing": plan.missing.strip(),
        "plan": plan.model_dump(),
    }