"""Whole-graph runs with a fake LLM: the paths, the loops, and their limits."""
from src.agent import config
from src.agent.answering import MaterialAnswer
from src.agent.classifier import IntentClassifier
from src.agent.critic import Review

EXPLAIN = IntentClassifier(message_intent="Explain", search_query="algorithm definition")
GOOD = MaterialAnswer(answer="An algorithm is a sequence of unambiguous instructions (book.pdf, page 31).",
                      missing="")
WRONG = MaterialAnswer(answer="An algorithm must always run in linear time (book.pdf, page 31).", missing="")
SUPPORTED = Review(supported=True, issues=[])
REJECTED = Review(supported=False, issues=["page 31 says nothing about linear time"])


def test_small_talk_skips_the_book(graph, ask, use_llm):
    use_llm({IntentClassifier: [IntentClassifier(message_intent="Chat", search_query="")]})
    path, state = ask(graph, "hi")
    assert path == ["classifier", "chat agent"]
    assert state["messages"][-1].content == "Hi! What are you studying today?"


def test_good_answer_is_searched_checked_reviewed_and_shown(graph, ask, use_llm, fake_book):
    use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [GOOD], Review: [SUPPORTED]})
    path, state = ask(graph, "what is an algorithm")
    assert path == ["classifier", "retrieve", "explanation agent", "validate", "critic", "finalize"]
    assert state["messages"][-1].content == GOOD.answer


def test_made_up_page_is_sent_back_with_feedback_and_fixed(graph, ask, use_llm, fake_book):
    made_up = MaterialAnswer(answer="An algorithm is a recipe (book.pdf, page 999).", missing="")
    llm = use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [made_up, GOOD], Review: [SUPPORTED]})
    path, state = ask(graph, "what is an algorithm")
    assert path.count("explanation agent") == 2 and state["failed_checks"] == 1
    second_try = llm.prompts_for("MaterialAnswer")[1][-1]["content"]
    assert "page 999" in second_try  # the writer was told exactly what to fix
    assert state["messages"][-1].content == GOOD.answer


def test_retry_loop_searches_again_and_stops_at_the_limit(graph, ask, use_llm, fake_book):
    partial = MaterialAnswer(answer="Step 1 (book.pdf, page 37).", missing="the remaining steps")
    use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [partial] * (config.MAX_RETRIES + 1),
             Review: [SUPPORTED]}, text="steps of algorithm design")
    path, state = ask(graph, "explain the steps of designing algorithms")
    assert path.count("rewrite query") == config.MAX_RETRIES
    assert state["retry_count"] == config.MAX_RETRIES
    assert state["search_query"] == "steps of algorithm design"  # the rewritten query was used


def test_not_covered_is_honest_and_skips_the_critic(graph, ask, use_llm, fake_book):
    not_covered = MaterialAnswer(answer=config.NOT_COVERED, missing="photosynthesis")
    use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [not_covered] * (config.MAX_RETRIES + 1)})
    path, state = ask(graph, "expain photosythiss")
    assert "critic" not in path
    assert state["messages"][-1].content == config.NOT_COVERED


def test_critic_rejection_is_sent_back_and_fixed(graph, ask, use_llm, fake_book):
    llm = use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [WRONG, GOOD], Review: [REJECTED, SUPPORTED]})
    path, state = ask(graph, "what is an algorithm")
    assert path == ["classifier", "retrieve", "explanation agent", "validate", "critic",
                    "explanation agent", "validate", "critic", "finalize"]
    assert "Reviewer: page 31 says nothing about linear time" in llm.prompts_for("MaterialAnswer")[1][-1]["content"]
    assert state["messages"][-1].content == GOOD.answer and state["review_count"] == 1


def test_critic_rejecting_twice_stops_and_warns(graph, ask, use_llm, fake_book):
    use_llm({IntentClassifier: [EXPLAIN], MaterialAnswer: [WRONG, WRONG], Review: [REJECTED, REJECTED]})
    path, state = ask(graph, "what is an algorithm")
    assert path.count("critic") == 2
    assert "could not confirm every claim" in state["messages"][-1].content
