"""Shared setup for every test: NO API key, NO network, NO real database.
- the LLM is replaced by FakeLLM (it gives back prepared answers)
- the book search is replaced by a fake search (it returns prepared pages)
- conversations are kept in memory, never in checkpoints.db
- the mock "share" action writes to a temporary file, never the real outbox"""
import os
import sys
import types

import pytest

# 1. before anything else is imported: no tracing, and a dummy key (the real model is never called)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

# 2. a fake book search, so tests never open Chroma or download an embedding model
fake_store = types.ModuleType("src.rag.vector_store")
fake_store.search = lambda query, num_of_results=4: []
fake_store.add_pdf = lambda pdf_bytes, source_name: 0
sys.modules["src.rag.vector_store"] = fake_store

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from src.agent import config  # noqa: E402

config.CHECKPOINT_DB = ":memory:"  # never touch the real checkpoints.db

from src.agent.graph import build_graph  # noqa: E402

# the two pages every fake search returns
BOOK = [
    {"text": "An algorithm is a sequence of unambiguous instructions for solving a problem.",
     "source": "book.pdf", "page": 31, "distance": 0.6},
    {"text": "Understanding the problem is the first step in designing an algorithm.",
     "source": "book.pdf", "page": 37, "distance": 0.7},
]
THREAD = {"configurable": {"thread_id": "test-thread"}}


class FakeLLM:
    """A pretend LLM: gives back prepared answers instead of calling a real model.
    `answers` maps each structured-output class to the answers to give, in order."""

    def __init__(self, answers=None, text="Hi! What are you studying today?"):
        self.answers = {schema: list(items) for schema, items in (answers or {}).items()}
        self.text = text
        self.prompts = []  # (schema name, messages) for every call, so tests can look inside

    def with_structured_output(self, schema):
        llm = self

        class Runner:
            def invoke(self, messages):
                llm.prompts.append((schema.__name__, messages))
                return llm.answers[schema].pop(0)

        return Runner()

    def invoke(self, messages):
        self.prompts.append(("text", messages))
        return AIMessage(content=self.text)

    def prompts_for(self, schema_name):
        """The prompts sent for one kind of answer, e.g. 'MaterialAnswer'."""
        return [messages for name, messages in self.prompts if name == schema_name]


@pytest.fixture
def use_llm(monkeypatch):
    """llm = use_llm({...}) swaps the real LLM for a FakeLLM, for one test only."""
    def _use(answers=None, **kwargs):
        fake = FakeLLM(answers, **kwargs)
        monkeypatch.setattr(config, "llm", fake)
        return fake
    return _use


@pytest.fixture
def fake_book(monkeypatch):
    """Every search returns the two pages in BOOK."""
    monkeypatch.setattr(fake_store, "search", lambda query, num_of_results=4: list(BOOK))


@pytest.fixture
def graph():
    """A fresh graph for each test, keeping conversations in memory only."""
    return build_graph(InMemorySaver())


@pytest.fixture
def ask():
    """path, state = ask(graph, "question") runs one question and records which nodes ran."""
    def _ask(graph, question):
        path = []
        for step in graph.stream({"messages": [{"role": "user", "content": question}]},
                                 THREAD, stream_mode="updates"):
            path += [node for node in step if node != "__interrupt__"]
        return path, graph.get_state(THREAD).values
    return _ask


@pytest.fixture
def outbox(tmp_path, monkeypatch):
    """The mock 'share' action writes to a temporary file instead of the real outbox."""
    path = tmp_path / "shared_plans.jsonl"
    monkeypatch.setattr(config, "OUTBOX_FILE", str(path))
    return path
