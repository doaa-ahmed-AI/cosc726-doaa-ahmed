"""Feature: explain and summarize, using only the course material."""
from pydantic import BaseModel, Field

from src.agent import config
from src.agent.retrieval import format_docs
from src.agent.state import StudyState
from src.agent.validation import feedback_text


class MaterialAnswer(BaseModel):
    answer: str = Field(description="The answer for the student, with citations.")
    missing: str = Field(
        description="The part of the question the material did not cover. Empty string if nothing is missing.")


def answer_from_material(state: StudyState, task: str):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    final_prompt = f"""Course material:
{context}

Question: {question}

Rules:
- Answer using the course material above. Do not add facts from your own knowledge.
- After every point, cite its source in the same format as the material headers, e.g. (lecture.pdf, page 12).
- It is fine if the material only covers part of the question. Explain the parts it covers,
  then say which parts were not found, and write those parts in the "missing" field.
  Example: if the question asks for all the steps and the material describes only 3,
  explain those 3, and write "the remaining steps of ..." in "missing".
- If nothing is missing, leave "missing" empty.
- Only if the material is about a completely different topic, set "answer" to exactly: {config.NOT_COVERED}
  and write the topic of the question in "missing".{feedback_text(state)}"""

    messages = ([{"role": "system", "content": task}]
                + history
                + [{"role": "user", "content": final_prompt}])

    structured_llm = config.llm.with_structured_output(MaterialAnswer)
    result = structured_llm.invoke(messages)

   
    return {"draft_answer": result.answer, "missing": result.missing.strip()}


def prompt_llm_explanation(state: StudyState):
    return answer_from_material(
        state, "You are a helpful tutor that explains concepts clearly to a university student.")


def prompt_llm_summary(state: StudyState):
    return answer_from_material(
        state, "You are a helpful assistant that summarizes course material.")