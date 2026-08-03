from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage, SystemMessage, AIMessage
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

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

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
@tool
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

# 4. date time 
@tool 
def get_current_datetime() -> dict:
    """Get the current date and time. Use this whenever the question depends
    on 'today', 'now', or the current date — never guess dates from memory."""
    return {"current datetime":datetime.now().strftime("%A, %d %B %Y, %I:%M %p")}

# 5. python REPL - Executes Python code dynamically to perform calculations and data analysis.
python_repl = PythonREPLTool()
@tool
def python_repl_tool(code: str) -> str:
    """
    Execute Python code for calculations, data analysis, and programming tasks.

    Use this tool ONLY when computation is required.
    Never use this tool to obtain live information such as:
    - Weather
    - Currency exchange rates
    - Stock prices
    - News
    - Current date or time

    For live information, always use the appropriate dedicated tool.
    """
    return python_repl.run(code)

# 6. wikipedia
@tool
def wikipedia_tool(query:str) -> str:
    """
    Search Wikipedia for established facts, historical or encyclopedic
    information. Use this for general knowledge questions about people,
    places, events, or concepts.
    """
    try:
        wiki = WikipediaAPIWrapper()
        result = wiki.run(query)
        if not result or not result.strip():
            return f"No Wikipedia results found for '{query}'."
        return result
    except Exception as e:
        return f"Wikipedia search failed (try rephrasing the query): {e}"

# 7. exchange rate. (eg, USD to INR)
@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the LIVE exchange rate between two currencies.

    Use this tool ONLY for currency conversion or exchange rates.
    Never use Python or web search instead."""
    try:
        url = "https://api.frankfurter.app/latest"
        params = {"from": from_currency.upper(), "to": to_currency.upper()}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()} (as of {data['date']})"
    except Exception as e:
        return f"Exchange rate error: {e}"


# make LLM tools aware...
tools = [search_tool, wikipedia_tool, python_repl_tool,get_stock_price, get_current_datetime, get_weather,get_exchange_rate]
llm_with_tools = llm.bind_tools(tools)

# ====================================== Chat Bot backend ======================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools: web search, wikipedia, "
    "stock price, weather, currency exchange rate, current date/time, and "
    "python for calculations. Answer general knowledge questions directly, "
    "in plain text — do NOT call a tool unless the question genuinely needs "
    "live/external/computed data.\n\n"
    "For ANY numeric fact that can change over time (prices, exchange rates, "
    "weather, stock values) — always call the relevant tool. Never guess, "
    "estimate, or recall such numbers from memory.\n\n"
    "When you need multiple tools to answer one question (e.g. converting a "
    "price to another currency), call them silently and give ONLY the final "
    "answer. Do NOT narrate your plan or steps out loud — never write things "
    "like 'first I will...', 'let me get...', or 'now I will calculate...'. "
    "The user should only see the final, direct answer.\n\n"
    "When a tool returns a result, NEVER show raw JSON or dict output to the "
    "user — always summarize it in a natural, clear sentence."
))

def chat_node(state: ChatState):
    messages = state['messages']

    full_message = [SYSTEM_PROMPT] + messages
    try:
        response = llm_with_tools.invoke(full_message)
    except Exception as e:
        print(f"chat_node error: {e}")   # shows the REAL cause in your terminal
        response = AIMessage(content=f"Something went wrong on my end: {e}")
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