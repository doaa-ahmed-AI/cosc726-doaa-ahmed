from dotenv import load_dotenv
import uuid
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


llm = init_chat_model('openai:gpt-6-luna')


def prompt_llm(state: MessagesState):
    response = llm.invoke(state['messages'])
    return {'messages': [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
graph.get_graph().print_ascii()

config = {'configurable': {'thread_id': uuid.uuid4()}}


while True:
    user_message = input('Enter message :')
    print(graph.invoke(
        {'messages': [{'role': 'user', 'content': user_message}]}, config=config))
