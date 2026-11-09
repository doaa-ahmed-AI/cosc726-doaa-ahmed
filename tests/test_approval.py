"""The human approval gate: pause, approve / edit / reject, and the guardrail on the action."""
import json

import pytest
from langgraph.types import Command

from conftest import THREAD
from src.agent.actions import share_study_plan
from src.agent.classifier import IntentClassifier
from src.agent.plan import StudyPlan, StudySession

PLAN = StudyPlan(sessions=[StudySession(day="Day 1", topic="What is an algorithm", page=31,
                                        activity="Read and summarize")], missing="")


@pytest.fixture
def paused_plan(graph, ask, use_llm, fake_book, outbox):
    """Ask for a plan; the graph pauses at the approval gate."""
    use_llm({IntentClassifier: [IntentClassifier(message_intent="Plan", search_query="algorithms")],
             StudyPlan: [PLAN]})
    ask(graph, "make me a study plan")
    return graph


def decide(graph, choice, plan="the plan"):
    return graph.invoke(Command(resume={"choice": choice, "plan": plan}), THREAD)


def test_plan_pauses_and_nothing_is_shared_yet(paused_plan, outbox):
    request = paused_plan.get_state(THREAD).interrupts[0].value
    assert request["action"] == "share_study_plan"
    assert "Day 1: What is an algorithm" in request["plan"]
    assert not outbox.exists()


def test_approve_shares_the_plan(paused_plan, outbox):
    result = decide(paused_plan, "approve")
    shared = [json.loads(line) for line in outbox.read_text().splitlines()]
    assert len(shared) == 1 and shared[0]["approval"] == "approve"
    assert "Shared with your study group" in result["messages"][-1].content


def test_edit_shares_the_humans_version(paused_plan, outbox):
    decide(paused_plan, "edit", plan="MY OWN PLAN: read page 31")
    shared = json.loads(outbox.read_text().splitlines()[0])
    assert shared["plan"] == "MY OWN PLAN: read page 31"


def test_reject_shares_nothing(paused_plan, outbox):
    result = decide(paused_plan, "reject")
    assert not outbox.exists()
    assert "Not shared" in result["messages"][-1].content


def test_a_typo_counts_as_reject(paused_plan, outbox):
    decide(paused_plan, "aprove")  # safe default: anything unclear is a "no"
    assert not outbox.exists()


def test_share_action_refuses_without_approval(outbox):
    with pytest.raises(PermissionError):
        share_study_plan({"approval": None, "draft_answer": "a plan"})
    assert not outbox.exists()
