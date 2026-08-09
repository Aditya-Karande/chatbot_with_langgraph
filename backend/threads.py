from langchain_core.messages import HumanMessage

from backend.config import checkpointer, LLM, pool


def init_thread_table():
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_metadata(
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )


init_thread_table()


def generate_title(first_msg: str) -> str:
    prompt = (
        "Turn this message into a short chat title (max 5 words). "
        "Reply with the title only, no quotes, no punctuation at the end:\n\n"
        f"{first_msg}"
    )
    response = LLM.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def save_thread(thread_id: str, title: str):
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO NOTHING",
            (str(thread_id), title),
        )


def get_all_threads():
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, title FROM thread_metadata ORDER BY created_at DESC"
        ).fetchall()
    return [{"thread_id": r[0], "title": r[1]} for r in rows]


def rename_title(thread_id: str, new_title: str):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE thread_metadata SET title = %s WHERE thread_id = %s",
            (new_title, str(thread_id)),
        )


def delete_thread(thread_id: str):
    with pool.connection() as conn:
        conn.execute("DELETE FROM thread_metadata WHERE thread_id = %s", (str(thread_id),))
    checkpointer.delete_thread(str(thread_id))
