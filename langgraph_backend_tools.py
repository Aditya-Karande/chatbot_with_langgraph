from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from datetime import datetime
import os
import requests

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

# ======================================= Tools =================================================

# 1. Duck Duck Go Search
search_tool = DuckDuckGoSearchRun(region="us-en")

# 2. Stock Price. (Alpha Vantage)
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=8NP3PSK04VUM3WAO"
    r = requests.get(url)
    return r.json()

# 3. weather 
def get_weather(city: str) -> dict:

    """Get the current real weather for a specific city. Use this whenever
    the question asks about current weather, temperature, or conditions
    somewhere — never guess weather from memory, it changes constantly."""

    api_key = os.environ["OPENWEATHER_API_KEY"]

    if not api_key:
        return {"error":"Weather tool not configured: missing OPENWEATHER_API_KEY."}

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q":city, "appid":api_key, "units":"metric"}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 404:
            return {"error":"city not found :("}

        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        condition = data["weather"][0]["description"]

        return {"city":city, "temperature":temp, "feels like":feels_like, "condition":condition}

    except Exception as e:
        return {"error": e}

# 3. date time 
@tool 
def get_current_datetime() -> dict:
    """Get the current date and time. Use this whenever the question depends
    on 'today', 'now', or the current date — never guess dates from memory."""
    return {"current datetime":datetime.now().strftime("%A, %d %B %Y, %I:%M %p")}

# 4. python REPL - Executes Python code dynamically to perform calculations and data analysis.
python_repl_tool = PythonREPLTool()

# 5. wikipedia
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


# make LLM tools aware...
tools = [search_tool, wikipedia_tool, get_stock_price, get_current_datetime, get_weather]
llm_with_tools = llm.bind_tools(tools)

# ====================================== Chat Bot backend ======================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        return {"error":e}
    return {"messages": [response]}

tools_node = ToolNode(tools=tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools","chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ================================== Helper functions for database ===================================

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