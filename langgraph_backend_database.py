from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import os

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# postgres connection.
DB_URL = os.environ["DATABASE_URL"]

connection_kwargs = {
    "autocommit":True,
    "prepare_threshold":0
}

# A pool (not a single connection) so multiple Streamlit reruns/users don't fight over one connection
pool = ConnectionPool(conninfo=DB_URL, max_size=20, kwargs=connection_kwargs)

# checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup() # creates LangGraph's internal tables — safe to call every run, it no-ops if they exist

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# own tabel for thread titles..
def init_thread_tabel():
    with pool.connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata(
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL, 
        created_at TIMESTAMP DEFAULT NOW()
    )
""")

init_thread_tabel()

def generate_title(first_msg: str) -> str:
    """one LLM call that converts user's first message into a simple chat title."""
    prompt = (
        "Turn this message into a short chat title (max 5 words). "
        "Reply with the title only, no quotes, no punctuation at the end:\n\n"
        f"{first_msg}"
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content.strip()

def save_thread(thread_id: str, title: str):
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO thread_metadata (thread_id, title) VALUES (%s, %s) "
            "ON CONFLICT (thread_id) DO NOTHING",
            (str(thread_id), title)
        )

def get_all_threads():
    """returns: thread_id and title in newest first order"""
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT thread_id, title FROM thread_metadata ORDER BY created_at DESC"
        ).fetchall()

    return [{"thread_id":r[0], "title": r[1]} for r in rows]

def rename_title(thread_id:str, new_title: str):
    with pool.connection() as conn:
        conn.execute(
            "UPDATE thread_metadata SET title = %s WHERE thread_id = %s ",
            (new_title, str(thread_id))
        )

def delete_thread(thread_id: str):
    # 1) remove from title row..
    with pool.connection() as conn:
        conn.execute("DELETE FROM thread_metadata WHERE thread_id = %s ",(str(thread_id),))

    # 2) remove the actual conversation state LangGraph stored for this state..
    checkpointer.delete_thread(str(thread_id))