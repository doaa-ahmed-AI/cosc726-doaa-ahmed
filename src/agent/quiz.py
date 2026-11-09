"""Feature: quizzes in a fixed format (4 options, a correct answer, an explanation, a page)."""
from typing import Literal

from pydantic import BaseModel, Field

from src.agent import config
from src.agent.retrieval import format_docs
from src.agent.state import StudyState
from src.agent.validation import feedback_text


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options, without letters.")
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str = Field(description="Why the correct option is right, based on the material.")
    page: int = Field(description="The page number (from the material headers) the question is based on.")


class Quiz(BaseModel):
    questions: list[QuizQuestion]
    missing: str = Field(description="What the material did not cover. Empty string if nothing is missing.")


def format_quiz(quiz: Quiz) -> str:
    """Turn the quiz data into readable text: questions first, answers at the end."""
    if not quiz.questions:
        return config.NOT_COVERED

    lines = []
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(f"**{i}. {q.question}**\n")
        for letter, option in zip("ABCD", q.options):
            lines.append(f"- {letter}) {option}")
        lines.append("")
    lines.append("**Answers**\n")
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(f"{i}. **{q.correct_option}**: {q.explanation} (page {q.page})")
    return "\n".join(lines)


def make_quiz(state: StudyState):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    prompt = f"""Course material:
{context}

Request: {question}

Rules:
- Write multiple-choice questions using ONLY the course material above.
- Each question has exactly 4 options and exactly one correct option (A, B, C or D).
- "page" is the page number (from the material headers) that the question is based on.
- "explanation" says why the correct option is right, using the material.
- If the request does not say how many questions, write 3.
- If the material covers only part of the request, write questions about that part and put the rest in "missing".
- If the material is about a completely different topic, return no questions and write the topic in "missing".{feedback_text(state)}"""

    messages = ([{"role": "system",
                  "content": "You are a helpful assistant that creates quizzes from course material."}]
                + history
                + [{"role": "user", "content": prompt}])

    structured_llm = config.llm.with_structured_output(Quiz)
    quiz = structured_llm.invoke(messages)

    return {
        "draft_answer": format_quiz(quiz),
        "missing": quiz.missing.strip(),
        "quiz": quiz.model_dump(),
    }