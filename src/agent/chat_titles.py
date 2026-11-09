"""Feature: a short title for each saved conversation, shown in the sidebar."""
import sqlite3
from datetime import datetime

from src.agent import config

conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS chat_titles (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    updated_at TEXT NOT NULL)""")
conn.commit()


def make_title(question: str, answer: str) -> str:
    """Ask the LLM for a 3-6 word title, based on the first question and answer."""
    prompt = f"""Write a short title (3 to 6 words) for a study conversation that starts like this:
Student: {question}
Assistant: {answer[:500]}
Reply with only the title, without quotes."""
    try:
        title = config.llm.invoke([{"role": "user", "content": prompt}]).content.strip().strip('"')
    except Exception:
        title = ""  
    return title[:60] or question[:40]


def get_title(thread_id: str) -> str | None:
    row = conn.execute("SELECT title FROM chat_titles WHERE thread_id = ?", (thread_id,)).fetchone()
    return row[0] if row else None


def save_title(thread_id: str, title: str):
    """Save a new chat's title, or mark an existing chat as just used (it keeps its title)."""
    conn.execute(
        """INSERT INTO chat_titles (thread_id, title, updated_at) VALUES (?, ?, ?)
           ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at""",
        (thread_id, title, datetime.now().isoformat()))
    conn.commit()


def list_chats(limit: int = 20) -> list[tuple[str, str]]:
    """The most recently used chats first: [(thread_id, title), ...]"""
    return conn.execute(
        "SELECT thread_id, title FROM chat_titles ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()