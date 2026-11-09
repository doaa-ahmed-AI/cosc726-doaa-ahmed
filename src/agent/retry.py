"""Feature: the retry loop - write a better search query for the part that is missing."""
from src.agent import config
from src.agent.state import StudyState


def rewrite_query(state: StudyState):
    question = state["messages"][-1].content
    prompt = f"""A student asked: {question}
We searched their textbook with this query: {state["search_query"]}
The answer is still missing: {state["missing"]}

Write a new short search query (a few keywords) that would find the missing part in a textbook.
Fix any spelling mistakes. Use different words from the previous query.
Reply with only the query."""

    response = config.llm.invoke([{"role": "user", "content": prompt}])
    return {
        "search_query": response.content.strip().strip('"'),
        "retry_count": state["retry_count"] + 1,
    }