"""The plain-Python checks (validation.py) and the text formats (quiz.py, plan.py)."""
from src.agent import config
from src.agent.plan import StudyPlan, format_plan
from src.agent.quiz import Quiz, QuizQuestion, format_quiz
from src.agent.validation import find_cited_pages, find_problems

FOUND = [{"page": 31}, {"page": 37}]  # the pages the search "found"


def test_finds_single_pages_and_ranges():
    assert find_cited_pages("(book.pdf, page 37); pages 39–40; Page 31") == {31, 37, 39, 40}


def test_good_answer_has_no_problems():
    assert find_problems("An algorithm is... (book.pdf, page 31)", FOUND, None) == []


def test_made_up_page_is_caught():
    problems = find_problems("Wrong (book.pdf, page 999)", FOUND, None)
    assert len(problems) == 1 and "page 999" in problems[0]


def test_answer_without_citations_is_caught():
    assert find_problems("An algorithm is a recipe.", FOUND, None) == [
        "The answer has no citations. Cite a page after every point."]


def test_not_covered_needs_no_citations():
    assert find_problems(config.NOT_COVERED, FOUND, None) == []


def test_quiz_with_three_options_is_caught():
    quiz = {"questions": [{"question": "q", "options": ["a", "b", "c"], "correct_option": "A",
                           "explanation": "e", "page": 31}]}
    assert find_problems("x (page 31)", FOUND, quiz) == ["Question 1 has 3 options; it must have exactly 4."]


def test_quiz_text_shows_options_and_answers():
    quiz = Quiz(questions=[QuizQuestion(question="What is an algorithm?", options=["a", "b", "c", "d"],
                                        correct_option="B", explanation="From the book.", page=31)], missing="")
    text = format_quiz(quiz)
    assert "- D) d" in text and "**Answers**" in text and "(page 31)" in text


def test_empty_quiz_and_empty_plan_mean_not_covered():
    assert format_quiz(Quiz(questions=[], missing="x")) == config.NOT_COVERED
    assert format_plan(StudyPlan(sessions=[], missing="x")) == config.NOT_COVERED
