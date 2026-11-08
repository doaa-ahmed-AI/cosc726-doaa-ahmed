import re
import uuid
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from src.rag.vector_store import search
from src.models.model import model


load_dotenv()


llm = init_chat_model(model=model.small_model)


MessageIntent = Literal["Chat", "Explain", "Summarize", "Quiz"]

NOT_COVERED = "This is not covered in your course material."
MAX_RETRIES = 2
MAX_FIXES = 1


# structerd output schemas
class IntentClassifier(BaseModel):
    message_intent: MessageIntent

    search_query: str = Field(
        description="A short, complete search query for the textbook (a few keywords).Empty for chat."
    )


class MaterialAnswer(BaseModel):
    answer: str = Field(
        description="The answer for the student , with citations.")
    missing: str = Field(
        description="The part of the question the material did not cover. Empty string of nothing is missing.")


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(
        description="Exactly 4 answer options, without letters.")
    correct_option: Literal["A", "B", "C", "D"]
    explanation: str = Field(
        description="Why the correct option is right, based on the material.")
    page: int = Field(
        description="The page number (from the material headers) the question is based on.")


class Quiz(BaseModel):
    questions: list[QuizQuestion]
    missing: str = Field(
        description="What the material did not cover. Empty string if nothing is missing.")


class StudyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: MessageIntent | None
    retrieved_docs: list[dict]
    draft_answer: str
    missing: str
    search_query: str
    retry_count: int
    quiz: dict | None
    problems: list[str]
    failed_checks: int


PAGE_PATTERN = re.compile(
    r"\bpages?\s+(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE)


def find_cited_pages(text: str) -> set[int]:
    pages = set()
    for start, end in PAGE_PATTERN.findall(text):
        pages.add(int(start))
        if end:
            pages.add(int(end))
    return pages


def find_problems(answer: str, docs: list[dict], quiz: dict | None) -> list[str]:
    if answer.strip() == NOT_COVERED:
        return []  # an honest "not covered" needs no citations

    problems = []
    found_pages = {d["page"] for d in docs}
    cited_pages = find_cited_pages(answer)

    if not cited_pages:
        problems.append(
            "The answer has no citations. Cite a page after every point.")
    for page in sorted(cited_pages - found_pages):
        problems.append(
            f"The answer cites page {page}, but that page is not in the course material you were given.")
    if quiz is not None:
        for i, q in enumerate(quiz["questions"], start=1):
            if len(q["options"]) != 4:
                problems.append(
                    f"Question {i} has {len(q['options'])} options; it must have exactly 4.")
            if not q["explanation"].strip():
                problems.append(f"Question {i} has no explanation.")

    return problems


def classify_intent(state: StudyState):
    structured_llm = llm.with_structured_output(IntentClassifier)
    question = state['messages'][-1].content
    recent = state['messages'][-7:-1]
    history_text = "\n".join(
        f"{m.type}: {m.content}" for m in recent) or "(no earlier messages)"

    result = structured_llm.invoke([
        {"role": "system",
         "content": """You are the router for a study assistant. Classify the user's latest message into exactly one category:
        - Explain: asks to explain, describe, or answer a question about a concept or topic
          (for example "what is X", "how does X work", "explain the steps of X").
        - Summarize: asks to summarize or shorten a topic or content.
        - Quiz: asks for a quiz, practice questions, or to test their knowledge.
        - Chat: ONLY greetings, thanks, small talk, or requests clearly unrelated to studying (like recipes).
        The message may contain spelling mistakes; focus on what the user means.
        If you are unsure between Chat and another category, choose the other category.
        If the mesaage refers to something earlier ("that ,"it") but there are nor earlier messages , choose Chat.
        Also write search_query: a short, complete search query for the textbook(a few keywords).
        -If the latest message refers to earlier messages ("that","it","explain more"),
        use the recent conversation to find the topic.
        -Fix spelling mistakes.
        _For Chat, leave search_query empty."""},
        {'role': "user",
         'content': f"Recent conversation:\n{history_text}\n\nLatest message:\n{question}"}
    ])

    return {"message_intent": result.message_intent,
            "search_query": result.search_query.strip() or question,
            "retrieved_docs": [],
            "draft_answer": "",
            "missing": "",
            "retry_count": 0,
            "quiz": None,
            "problems": [],
            "failed_checks": 0
            }


def retrieve(state: StudyState):
    new_docs = search(state["search_query"], num_of_results=6)
    docs = list(state["retrieved_docs"])
    seen_texts = {d["text"] for d in docs}
    for doc in new_docs:
        if doc["text"] not in seen_texts:
            docs.append(doc)
            seen_texts.add(doc["text"])
    return {"retrieved_docs": docs}


def format_docs(docs: list[dict]) -> str:
    if not docs:
        return "No course material found."

    return "\n\n".join(
        f"[{d['source']},page {d['page']}]\n{d['text']}" for d in docs)


def feedback_text(state: StudyState) -> str:
    if not state.get("problems"):
        return ""
    lines = "\n".join(f"- {p}" for p in state["problems"])
    return f"\n\nYour previous answer had these problems. Fix them:\n{lines}"


def answer_from_material(state: StudyState, task: str):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    final_prompt = f"""Course material:
{context}
 
Question: {question}
 
Rules:
- Answer using the course material. Do not add facts from your own knowledge.
- After every point, cite its source in the same format as the material headers, e.g. (lecture.pdf, page 12).
- It is fine is the material only covers part of the question, Explain the parts it covers,
then say which parts were not found, and write those parts in the "missing" field.
Example: if the question asks for all the steps and the material describes only 3,
explain those 3, and write "the remaining steps of ... " in "missing.
-If nothing is missing, leave "missing" empty.
-Only if the material is about a completely diffrent topic, set "answer" to exactly: {NOT_COVERED}
and write the topic of the question in "missing"."""

    messages = ([{"role": "system", "content": task}]
                + history
                + [{"role": "user", "content": final_prompt}])

    structured_llm = llm.with_structured_output(MaterialAnswer)
    result = structured_llm.invoke(messages)

    return {"draft_answer": result.answer, "missing": result.missing.strip()}


def format_quiz(quiz: Quiz) -> str:
    if not quiz.questions:
        return NOT_COVERED

    lines = []
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(f"**{i}. {q.question}**\n")
        for letter, option in zip("ABCD", q.options):
            lines.append(f"- {letter}) {option}")
        lines.append("")
    lines.append("**Answers**\n")
    for i, q in enumerate(quiz.questions, start=1):
        lines.append(
            f"{i}. **{q.correct_option}**: {q.explanation} (page {q.page})")
    return "\n".join(lines)


def make_quiz(state: StudyState):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    prompt = f"""Course material:
{context}
 
Request: {question}
 
Rules:
- Write multiple-choice questions using ONLY the course material.
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

    structured_llm = llm.with_structured_output(Quiz)
    quiz = structured_llm.invoke(messages)

    return {
        "draft_answer": format_quiz(quiz),
        "missing": quiz.missing.strip(),
        "quiz": quiz.model_dump(),
    }


def validate(state: StudyState):
    problems = find_problems(
        state["draft_answer"], state["retrieved_docs"], state.get("quiz"))
    failed_checks = state["failed_checks"] + (1 if problems else 0)
    return {"problems": problems, "failed_checks": failed_checks}


def rewrite_query(state: StudyState):
    question = state["messages"][-1].content
    prompt = f"""A student asked: {question}
We searched their textbook with this query: {state["search_query"]}
The answer is still missing: {state["missing"]}
 
Write a new short search query (a few keywords) that would find the missing part in a textbook.
Fix any spelling mistakes. Use different words from the previous query.
Reply with only the query."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return {
        "search_query": response.content.strip().strip('"'),
        "retry_count": state["retry_count"] + 1,
    }


def finalize(state: StudyState):
    return {"messages": [{"role": "assistant", "content": state["draft_answer"]}]}


def prompt_llm_chat(state: StudyState):
    messages = [{"role": "system",
                 "content": "You are a friendly study assistant. Reply to greetings and small talk briefly. "
                            "If the user asks for something unrelated to studying (like recipes), "
                            "politely say you are here to help with their course material."}] + state["messages"]
    response = llm.invoke(messages)
    return {'messages': [response]}


def prompt_llm_explanation(state: StudyState):
    return answer_from_material(
        state, "You are a helpful tutor that explains concepts clearly to a university student.")


def prompt_llm_summary(state: StudyState):
    return answer_from_material(
        state, "You are a helpful assistant that summarizes course material.")


def prompt_llm_quiz(state: StudyState):
    return answer_from_material(
        state, "You are a helpful assistant that creates quizzes from course material.")


# routing

AGENT_FOR_INTENT = {
    "Explain": "explanation agent",
    "Summarize": "summary agent",
    "Quiz": "quiz agent",
}


def route_after_classifier(state: StudyState) -> str:
    if state['message_intent'] == "Chat":
        return 'chat agent'
    return "retrieve"


def route_after_validate(state: StudyState) -> str:
    # 1. the answer broke a rule -> send it back to the same agent, with the problems
    if state["problems"] and state["failed_checks"] <= MAX_FIXES:
        return AGENT_FOR_INTENT[state["message_intent"]]
    # 2. something is missing -> search again
    if state["missing"] and state["retry_count"] < MAX_RETRIES:
        return "rewrite query"
    # 3. done
    return "finalize"


# graph
graph_builder = StateGraph(StudyState)
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('chat agent', prompt_llm_chat)
graph_builder.add_node('explanation agent', prompt_llm_explanation)
graph_builder.add_node('summary agent', prompt_llm_summary)
graph_builder.add_node('quiz agent', make_quiz)
graph_builder.add_node('rewrite query', rewrite_query)
graph_builder.add_node('validate', validate)
graph_builder.add_node('finalize', finalize)


graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges(
    "classifier", route_after_classifier, ["chat agent", "retrieve"])

graph_builder.add_conditional_edges(
    'retrieve', lambda state: state['message_intent'], {
        'Explain': 'explanation agent',
        'Summarize': 'summary agent',
        'Quiz': 'quiz agent'})


for node in ["explanation agent", "summary agent", "quiz agent"]:
    graph_builder.add_edge(node, 'validate')


graph_builder.add_conditional_edges(
    'validate', route_after_validate, ['explanation agent', 'summary agent', 'quiz agent', 'rewrite query', 'finalize'])

graph_builder.add_edge('rewrite query', 'retrieve')
graph_builder.add_edge('finalize', END)
graph_builder.add_edge('chat agent', END)


checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {'configurable': {'thread_id': str(uuid.uuid4())}}

    while True:
        user_message = input("Enter message:")
        result = graph.invoke(
            {'messages': [{'role': 'user', 'content': user_message}]}, config=config)

        print(result['messages'][-1].content)

        if result["message_intent"] != "Chat":
            pages = [
                f"{d['source']} p.{d['page']} ({d['distance']:.2f})" for d in result["retrieved_docs"]]
            print("sources used:", ",".join(pages))
            print("missing:", result["missing"] or "(nothing)")
            print(
                f"retries: {result['retry_count']} | last search query: {result['search_query']}")
            print(
                f"failed checks: {result['failed_checks']} | problems: {result['problems'] or '(none)'}")
