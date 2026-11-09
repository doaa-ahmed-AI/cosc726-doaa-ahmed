"""The map: puts every feature together into one graph, and saves conversations in SQLite.
Read this file to see the whole system at a glance."""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent import config
from src.agent.actions import share_study_plan 
from src.agent.answering import prompt_llm_explanation, prompt_llm_summary
from src.agent.approval import ask_approval
from src.agent.chat import prompt_llm_chat

from src.agent.classifier import classify_intent
from src.agent.finalize import finalize
from src.agent.plan import make_plan
from src.agent.quiz import make_quiz
from src.agent.critic import critic 
from src.agent.retrieval import retrieve
from src.agent.retry import rewrite_query
from src.agent.routing import (route_after_approval,route_after_critic, route_after_classifier, route_after_retrieve, route_after_validate)
from src.agent.state import StudyState
from src.agent.validation import validate




def build_graph(checkpointer):
    builder = StateGraph(StudyState)

    # the nodes (one per feature)
    builder.add_node("classifier", classify_intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("chat agent", prompt_llm_chat)
    builder.add_node("explanation agent", prompt_llm_explanation)
    builder.add_node("summary agent", prompt_llm_summary)
    builder.add_node("quiz agent", make_quiz)
    builder.add_node("plan agent", make_plan) 
    builder.add_node("ask approval", ask_approval)
    builder.add_node("share plan", share_study_plan) 
    builder.add_node("validate", validate)
    builder.add_node("critic", critic)  
    builder.add_node("rewrite query", rewrite_query)
    builder.add_node("finalize", finalize)

    # the arrows
    builder.add_edge(START, "classifier")
    builder.add_conditional_edges("classifier", route_after_classifier, ["chat agent", "retrieve"])
    builder.add_conditional_edges("retrieve", route_after_retrieve,
                                  ["explanation agent", "summary agent", "quiz agent","plan agent"])

    # every material answer goes through the checks first
    for node in ["explanation agent", "summary agent", "quiz agent","plan agent"]:
        builder.add_edge(node, "validate")

    builder.add_conditional_edges("validate", route_after_validate,
                                  ["explanation agent", "summary agent", "quiz agent","plan agent",
                                   "rewrite query", "ask approval", "critic", "finalize"])
    
    builder.add_conditional_edges("critic", route_after_critic,
                                  ["explanation agent", "summary agent", "quiz agent", "finalize"])

    
    builder.add_conditional_edges("ask approval", route_after_approval, ["share plan", "finalize"])
    builder.add_edge("share plan", "finalize")
    

    builder.add_edge("rewrite query", "retrieve")  # this arrow makes the search loop
    builder.add_edge("finalize", END)
    builder.add_edge("chat agent", END)            # chat goes straight to the end

    return builder.compile(checkpointer=checkpointer)


# save every step of every conversation in a SQLite file, so conversations survive a restart.
# check_same_thread=False: Streamlit runs our code from different threads.
conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
graph = build_graph(SqliteSaver(conn))