"""Evaluation harness (PHASE 4 - STEP 2).

Runs the task set in evals/tasks.json on three systems and prints the metrics:
  1. plain LLM     - no book, no search                                      (baseline 1)
  2. search once   - one search + one answer; no retries, checks or critic   (baseline 2)
  3. full agent    - the LangGraph agent
Uses your REAL model and your REAL course material (the book must already be added).

Run from the project folder:
  uv run python -m evals.run_eval            # everything
  uv run python -m evals.run_eval --quick    # first 3 tasks, agent only (a cheap check)

Each run is saved in evals/results/: a JSON file with every detail, and latest.md for the report.
"""
import argparse
import json
import os
import statistics
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

os.environ["LANGSMITH_PROJECT"] = "study-assistant-eval"  # keep evaluation traces separate in LangSmith

from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.agent import config  # noqa: E402
from src.agent.answering import answer_from_material  # noqa: E402
from src.agent.graph import build_graph  # noqa: E402
from src.agent.validation import find_cited_pages  # noqa: E402
from src.rag import vector_store  # noqa: E402

EVAL_DIR = Path(__file__).parent
SYSTEMS = ["plain LLM", "search once", "full agent"]
TUTOR = "You are a helpful tutor that explains concepts clearly to a university student."

# Fill these in from your model provider's pricing page to see the cost in dollars
# (dollars per 1 million tokens). Leave them as None to report tokens only.
PRICE_PER_1M_INPUT_TOKENS = None
PRICE_PER_1M_OUTPUT_TOKENS = None


# ---------- the three systems ----------

def plain_llm(task):
    """Baseline 1: just ask the LLM. No book, no search, no checks."""
    reply = config.llm.invoke([{"role": "system", "content": TUTOR},
                               {"role": "user", "content": task["question"]}])
    return {"answer": reply.content, "docs": [], "route": None}


def search_once(task):
    """Baseline 2: one search with the raw question, one answer (same rules prompt as the agent).
    No classifier, no retries, no validation, no critic, no chat history."""
    docs = vector_store.search(task["question"], num_of_results=config.NUM_RESULTS)
    state = {"messages": [HumanMessage(content=task["question"])], "retrieved_docs": docs,
             "problems": [], "review_issues": []}
    return {"answer": answer_from_material(state, TUTOR)["draft_answer"], "docs": docs, "route": None}


def full_agent(task, graph, thread):
    """The LangGraph agent. If it pauses for approval (a plan), the evaluation always rejects."""
    result = graph.invoke({"messages": [{"role": "user", "content": task["question"]}]}, thread)
    paused = "__interrupt__" in result
    if paused:
        result = graph.invoke(Command(resume={"choice": "reject", "plan": ""}), thread)
    quiz = result.get("quiz")
    return {
        "answer": result["messages"][-1].content,
        "docs": result.get("retrieved_docs", []),
        "route": result["message_intent"],
        "paused_for_approval": paused,
        "retries": result.get("retry_count", 0),
        "failed_checks": result.get("failed_checks", 0),
        "critic_rejections": result.get("review_count", 0),
        "quiz_valid": (quiz is not None and len(quiz["questions"]) > 0
                       and all(len(q["options"]) == 4 for q in quiz["questions"])
                       and not result.get("problems")),
    }


# ---------- measuring ----------

class CallCounter(BaseCallbackHandler):
    """Counts how many times an LLM is called."""

    def __init__(self):
        self.calls = 0

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.calls += 1

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.calls += 1


def measure(system, *args):
    """Run one system on one task, and record its time, LLM calls and tokens."""
    counter = CallCounter()
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        out = RunnableLambda(lambda _: system(*args)).invoke(None, config={"callbacks": [counter]})
    out["seconds"] = round(time.perf_counter() - start, 2)
    out["llm_calls"] = counter.calls
    out["input_tokens"] = sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values())
    out["output_tokens"] = sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())
    return out


def normalize(text):
    """Lowercase, and turn PDF ligatures like 'ﬁ' into normal letters."""
    return unicodedata.normalize("NFKC", text).lower()


def score(task, out):
    """The quality metrics for one answer."""
    answer = out["answer"]
    declined = answer.strip() == config.NOT_COVERED or out.get("route") == "Chat"
    cited = find_cited_pages(answer)
    found = {d["page"] for d in out["docs"]}
    keywords = task.get("keywords", [])
    return {
        "declined": declined,
        "valid_citations": bool(cited) and cited <= found and not declined,
        "cites_correct_page": bool(cited & set(task.get("gold_pages", []))),
        "keyword_recall": (sum(normalize(k) in normalize(answer) for k in keywords) / len(keywords)
                           if keywords else None),
    }


# ---------- summary ----------

def ratio(flags):
    flags = list(flags)
    if not flags:
        return "-"
    return f"{sum(flags)}/{len(flags)} ({100 * sum(flags) / len(flags):.0f}%)"


def average(values, digits=1):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), digits) if values else "-"


def cost(results):
    if PRICE_PER_1M_INPUT_TOKENS is None or PRICE_PER_1M_OUTPUT_TOKENS is None:
        return "set prices in run_eval.py"
    dollars = sum(r["input_tokens"] * PRICE_PER_1M_INPUT_TOKENS + r["output_tokens"] * PRICE_PER_1M_OUTPUT_TOKENS
                  for r in results) / 1_000_000
    return f"${dollars:.4f}"


def summarize(rows):
    """One column per system, compared on the same questions (in-scope + out-of-scope)."""
    comparison = {}
    for name in SYSTEMS:
        pairs = [(row["task"], row["results"][name]) for row in rows if name in row["results"]
                 and row["task"]["type"] in ("in_scope", "out_of_scope")]
        if not pairs:
            continue
        in_scope = [r for t, r in pairs if t["type"] == "in_scope"]
        out_scope = [r for t, r in pairs if t["type"] == "out_of_scope"]
        results = [r for _, r in pairs]
        comparison[name] = {
            "in-scope questions answered": ratio(not r["declined"] for r in in_scope),
            "answers with valid citations": ratio(r["valid_citations"] for r in in_scope),
            "answers citing a correct page": ratio(r["cites_correct_page"] for r in in_scope),
            "keyword recall (completeness)": f"{100 * average([r['keyword_recall'] for r in in_scope], 3):.0f}%"
                                             if in_scope else "-",
            "out-of-scope honestly declined": ratio(r["declined"] for r in out_scope),
            "avg seconds per question": average(r["seconds"] for r in results),
            "avg LLM calls per question": average(r["llm_calls"] for r in results),
            "avg tokens per question": average((r["input_tokens"] + r["output_tokens"] for r in results), 0),
            "total cost": cost(results),
        }

    agent = [(row["task"], row["results"]["full agent"]) for row in rows]
    agent_only = {
        "routing accuracy (all tasks)": ratio(r["route_ok"] for _, r in agent),
        "quiz in a valid format": ratio(r["quiz_valid"] for t, r in agent if t["type"] == "quiz"),
        "plan paused for human approval": ratio(r["paused_for_approval"] for t, r in agent if t["type"] == "plan"),
        "chat/off-topic kept out of the book": ratio(r["route"] == "Chat" for t, r in agent if t["type"] == "chat"),
        "avg retries (extra searches)": average(r["retries"] for _, r in agent),
        "answers rejected by the critic": sum(r["critic_rejections"] > 0 for _, r in agent),
        "answers that failed the checks": sum(r["failed_checks"] > 0 for _, r in agent),
        "avg seconds per task (all tasks)": average(r["seconds"] for _, r in agent),
        "total cost (all tasks)": cost([r for _, r in agent]),
    }
    return comparison, agent_only


def problems_to_look_at(rows):
    """Tasks where the agent did not do what we expected: material for the 'failure modes' section."""
    notes = []
    for row in rows:
        t, r = row["task"], row["results"]["full agent"]
        if not r["route_ok"]:
            notes.append(f"{t['id']}: routed to {r['route']}, expected {' or '.join(t['expected_route'])}")
        if t["type"] == "in_scope":
            if r["declined"]:
                notes.append(f"{t['id']}: declined a question the book covers")
            elif not r["valid_citations"]:
                notes.append(f"{t['id']}: citations missing or not from the searched pages")
            if not r["cites_correct_page"]:
                notes.append(f"{t['id']}: did not cite any of the expected pages {t['gold_pages']}")
            if r["keyword_recall"] is not None and r["keyword_recall"] < 0.5:
                notes.append(f"{t['id']}: incomplete (keyword recall {100 * r['keyword_recall']:.0f}%)")
        if t["type"] == "out_of_scope" and not r["declined"]:
            notes.append(f"{t['id']}: answered a question the book does not cover")
        if t["type"] == "quiz" and not r["quiz_valid"]:
            notes.append(f"{t['id']}: quiz not in a valid format")
        if t["type"] == "plan" and not r["paused_for_approval"]:
            notes.append(f"{t['id']}: did not pause for approval")
    return notes


def table(metrics_by_column, first_header):
    columns = list(metrics_by_column)
    metrics = list(next(iter(metrics_by_column.values())))
    lines = [f"| {first_header} | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for m in metrics:
        lines.append(f"| {m} | " + " | ".join(str(metrics_by_column[c][m]) for c in columns) + " |")
    return "\n".join(lines)


def per_task_table(rows):
    lines = ["| task | route OK | answered | valid citations | correct page | keyword recall "
             "| retries | critic rejections | seconds | LLM calls |", "|---" * 10 + "|"]
    for row in rows:
        t, r = row["task"], row["results"]["full agent"]
        recall = "-" if r["keyword_recall"] is None else f"{100 * r['keyword_recall']:.0f}%"
        lines.append(f"| {t['id']} | {'✅' if r['route_ok'] else '❌'} | {'no' if r['declined'] else 'yes'} "
                     f"| {'✅' if r['valid_citations'] else '-'} | {'✅' if r['cites_correct_page'] else '-'} "
                     f"| {recall} | {r['retries']} | {r['critic_rejections']} | {r['seconds']} | {r['llm_calls']} |")
    return "\n".join(lines)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the study assistant.")
    parser.add_argument("--quick", action="store_true", help="first 3 tasks, agent only")
    args = parser.parse_args()

    if vector_store.collection.count() == 0:
        print("Your course material is empty. Add the book first (in the app, or run src/rag/vector_store.py).")
        return

    tasks = json.loads((EVAL_DIR / "tasks.json").read_text(encoding="utf-8"))
    if args.quick:
        tasks = tasks[:3]
    graph = build_graph(InMemorySaver())  # evaluation chats stay in memory, never in checkpoints.db

    rows = []
    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] {task['id']}: {task['question']}")
        thread = {"configurable": {"thread_id": f"eval-{task['id']}-{uuid.uuid4().hex[:8]}"}}
        for earlier in task.get("history", []):  # set up the conversation first (not measured)
            graph.invoke({"messages": [{"role": "user", "content": earlier}]}, thread)

        results = {}
        out = measure(full_agent, task, graph, thread)
        results["full agent"] = out | score(task, out) | {"route_ok": out["route"] in task["expected_route"]}
        if not args.quick and task["type"] in ("in_scope", "out_of_scope"):
            for name, system in (("plain LLM", plain_llm), ("search once", search_once)):
                out = measure(system, task)
                results[name] = out | score(task, out)

        for r in results.values():  # keep only the page numbers, not the full text, in the saved file
            r["pages_found"] = sorted({d["page"] for d in r.pop("docs")})
        rows.append({"task": task, "results": results})

    comparison, agent_only = summarize(rows)
    notes = problems_to_look_at(rows)
    report = "\n\n".join([
        f"# Evaluation results ({datetime.now():%Y-%m-%d %H:%M}, {len(tasks)} tasks)",
        "## Agent vs. baselines (same in-scope + out-of-scope questions)",
        table(comparison, "metric") if comparison else "(baselines skipped in --quick mode)",
        "## Agent only (all tasks)",
        table({"full agent": agent_only}, "metric"),
        "## Agent, task by task",
        per_task_table(rows),
        "## Problems to look at",
        "\n".join(f"- {n}" for n in notes) or "- none",
    ])
    print("\n" + report)

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (results_dir / f"run-{stamp}.json").write_text(
        json.dumps({"comparison": comparison, "agent_only": agent_only, "problems": notes, "rows": rows},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "latest.md").write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved: evals/results/run-{stamp}.json and evals/results/latest.md")


if __name__ == "__main__":
    main()
