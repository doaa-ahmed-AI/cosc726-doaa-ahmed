"""Every decision in routing.py: which node runs next."""
from src.agent import config
from src.agent.routing import (route_after_approval, route_after_classifier, route_after_critic,
                               route_after_validate)

READY = {"message_intent": "Explain", "problems": [], "failed_checks": 0, "missing": "",
         "retry_count": 0, "plan": None, "draft_answer": "An algorithm is... (page 31)"}


def test_small_talk_goes_to_chat_and_study_questions_search_first():
    assert route_after_classifier({"message_intent": "Chat"}) == "chat agent"
    assert route_after_classifier({"message_intent": "Explain"}) == "retrieve"


def test_answer_that_breaks_a_rule_is_sent_back_only_once():
    assert route_after_validate({**READY, "problems": ["bad page"], "failed_checks": 1}) == "explanation agent"
    # after its one fix it moves on (finalize adds a warning at the end)
    assert route_after_validate({**READY, "problems": ["bad page"], "failed_checks": 2}) == "critic"


def test_missing_part_triggers_a_new_search_until_the_limit():
    assert route_after_validate({**READY, "missing": "the rest"}) == "rewrite query"
    assert route_after_validate({**READY, "missing": "the rest", "retry_count": config.MAX_RETRIES}) == "critic"


def test_finished_answer_goes_to_the_critic_but_not_covered_skips_it():
    assert route_after_validate(READY) == "critic"
    assert route_after_validate({**READY, "draft_answer": config.NOT_COVERED}) == "finalize"


def test_ready_plan_goes_to_the_human_not_the_critic():
    plan_ready = {**READY, "message_intent": "Plan", "plan": {"sessions": [{"day": "Day 1"}]}}
    assert route_after_validate(plan_ready) == "ask approval"


def test_critic_rejection_goes_back_only_once():
    assert route_after_critic({"message_intent": "Quiz", "review_issues": ["x"], "review_count": 1}) == "quiz agent"
    assert route_after_critic({"message_intent": "Quiz", "review_issues": ["x"], "review_count": 2}) == "finalize"
    assert route_after_critic({"message_intent": "Quiz", "review_issues": [], "review_count": 0}) == "finalize"


def test_only_approved_or_edited_plans_are_shared():
    assert route_after_approval({"approval": "approve"}) == "share plan"
    assert route_after_approval({"approval": "edit"}) == "share plan"
    assert route_after_approval({"approval": "reject"}) == "finalize"
