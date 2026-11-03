import uuid
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from src.models.model import model

load_dotenv()


llm = init_chat_model(model=model.small_model)


# structerd output schema for intent classification
class IntentClassifier(BaseModel):
    message_intent: Literal["Chat", "Explain", "Summarize", "Quiz"]

MessageIntent = Literal[
    "Chat",
    "Explain",
    "Summarize",
    "Quiz",
]
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    message_intent: str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {'role': "system",
            'content': '''Classify the user's latest message into exactly one category:
        - Chat: greetings, casual conversation, or general questions.
        - Explain: requests to explain a concept or text.
        - Summarize: requests to shorten or summarize content.
        - Quiz: requests to generate a quiz or test knowledge.
        If the message is a greeting or casual conversation,
        choose Chat.'''
         },
        {'role': "user", 'content': state['messages'][-1].content}
    ])

    return {"message_intent": result.message_intent}


def prompt_llm_chat(state: State):
    messages = [{'role': "system",
                 'content': 'You are a talkative chatbot be nice.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def prompt_llm_explanation(state: State):
    messages = [{'role': "system",
                 'content': 'You are a helpful assistant that explains text.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


def prompt_llm_summary(state: State):
    messages = [{'role': "system",
                 'content': 'You are a helpful assistant that summarizes text.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


# def prompt_llm_reviewer(state: State):
#     messages = [{'role': "system",
#                  'content': 'You are a helpful assistant that reviews the summarized text.'}] + state['messages']
#     response = llm.invoke(messages)
#     return {'messages': [{'role': 'assistant', 'content': response}]}


def prompt_llm_quiz(state: State):
    messages = [{'role': "system",
                 'content': 'You are a helpful assistant that creates quizzes.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


graph_builder = StateGraph(State)
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat agent', prompt_llm_chat)
graph_builder.add_node('explanation agent', prompt_llm_explanation)
graph_builder.add_node('summary agent', prompt_llm_summary)
# graph_builder.add_node('reviewer agent', prompt_llm_reviewer)
graph_builder.add_node('quiz agent', prompt_llm_quiz)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {
                                    'Chat': 'chat agent', 'Explain': 'explanation agent', 'Summarize': 'summary agent', 'Quiz': 'quiz agent'})
graph_builder.add_edge('chat agent', END)
graph_builder.add_edge('explanation agent', END)
graph_builder.add_edge('summary agent', END)
# graph_builder.add_edge('reviewer agent', END)
graph_builder.add_edge('quiz agent', END)


checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
config = {'configurable': {'thread_id': uuid.uuid4()}}

while True:
    user_message = input("Enter message:")
    result = graph.invoke(
        {'messages': [{'role': 'user', 'content': user_message}]}, config=config)

    print(result['messages'][-1].content)
