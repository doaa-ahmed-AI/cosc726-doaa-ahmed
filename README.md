# StudyU - Agent System

## 1. What it does

StudyU is an AI study assistant that helps students learn from their own course materials. Students upload lecture documents, and StudyU answers questions in that content, writes summaries, generates  quizzes, evaluates the student's answers, and builds a personalized study plan. It is built with Python, LangGraph and Supabase as a stateful agentic workflow:

for now it is a command-line chat loop over a single LangGraph node with in-session memory, and the retrieval, quizzing, planning, validation and approval features above are still to be built.

## 2. Prerequisites

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- Model runtime: hosted OpenAI API, model `gpt-6-luna` (set in `main.py`), with a valid API key

## 3. Install

```bash
git clone https://github.com/doaa-ahmed-AI/cosc726-doaa-ahmed.git
cd cosc726-doaa-ahmed
uv sync
```

## 4. Configure

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your real OpenAI key:

```
OPENAI_API_KEY=sk-...
```

## 5. Run

```bash
uv run main.py
```

When it works, you will see an ASCII drawing of the graph (`START -> prompt_llm -> END`), followed by the prompt `Enter message :`. Type a message and press enter; the model's reply appears inside the printed state dict. Quit with `Ctrl+C`.
