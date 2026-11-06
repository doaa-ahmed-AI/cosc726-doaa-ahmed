import uuid
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from src.rag.vector_store import search
from src.models.model import model


load_dotenv()


llm = init_chat_model(model=model.small_model)


MessageIntent = Literal["Chat", "Explain", "Summarize", "Quiz"]


# structerd output schema for intent classification
class IntentClassifier(BaseModel):
    message_intent: MessageIntent


class StudyState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: MessageIntent | None
    retrieved_docs: list[dict]


def classify_intent(state: StudyState):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {"role": "system",
         "content": """You are the router for a study assistant. Classify the user's latest message into exactly one category:
        - Explain: asks to explain, describe, or answer a question about a concept or topic
          (for example "what is X", "how does X work", "explain the steps of X").
        - Summarize: asks to summarize or shorten a topic or content.
        - Quiz: asks for a quiz, practice questions, or to test their knowledge.
        - Chat: ONLY greetings, thanks, small talk, or requests clearly unrelated to studying (like recipes).
        The message may contain spelling mistakes; focus on what the user means.
        If you are unsure between Chat and another category, choose the other category."""},
        {'role': "user", 'content': state['messages'][-1].content}
    ])

    return {"message_intent": result.message_intent}


def retrieve(state: StudyState):
    question = state["messages"][-1].content
    docs = search(question, num_of_results=4)
    return {"retrieved_docs": docs}


def format_docs(docs: list[dict]) -> str:
    if not docs:
        return "No course material found."

    return "\n\n".join(
        f"[{d['source']},page {d['page']}]\n{d['text']}" for d in docs)


NOT_COVERED = "This is not covered in your course material."


def answer_from_material(state: StudyState, task: str):
    context = format_docs(state["retrieved_docs"])
    question = state["messages"][-1].content
    history = state["messages"][:-1]

    final_prompt = f"""Course material:
{context}
 
Question: {question}
 
Rules:
- Use ONLY the course material. Do not add facts from your own knowledge.
- After every point, cite its source in the same format as the material headers, e.g. (lecture.pdf, page 12).
- If the material does not answer the question, reply exactly: {NOT_COVERED}"""

    messages = ([{"role": "system", "content": task}]
                + history
                + [{"role": "user", "content": final_prompt}])
    response = llm.invoke(messages)
    return {"messages": [response]}


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

def route_after_classifier(state: StudyState) -> str:
    if state['message_intent'] == "Chat":
        return 'chat agent'
    return "retrieve"


# graph
graph_builder = StateGraph(StudyState)
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('chat agent', prompt_llm_chat)
graph_builder.add_node('explanation agent', prompt_llm_explanation)
graph_builder.add_node('summary agent', prompt_llm_summary)
graph_builder.add_node('quiz agent', prompt_llm_quiz)


graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges(
    "classifier", route_after_classifier, ["chat agent", "retrieve"])

graph_builder.add_conditional_edges(
    'retrieve', lambda state: state['message_intent'], {
        'Explain': 'explanation agent',
        'Summarize': 'summary agent',
        'Quiz': 'quiz agent'})


for node in ["chat agent", "explanation agent", "summary agent", "quiz agent"]:
    graph_builder.add_edge(node, END)


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
