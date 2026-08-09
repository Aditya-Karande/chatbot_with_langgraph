import os
import requests
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool

# tool 1 - Duck Duck Go Search (web search)
search_tool = DuckDuckGoSearchRun(region="us-en")

# tool 2 - Stock Price.
@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for given symbol (e.g. 'AAPL','TSLA')"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=8NP3PSK04VUM3WAO"
    r = requests.get(url)
    return r.json()

# tool 3 - weather.
@tool
def get_weather(city:str) -> dict:
    """Get the current real weather for a specific city. Never guess weather
    from memory, it changes constantly."""
    api_key = os.environ["OPENWEATHER_API_KEY"]

    if not api_key:
        return{"error":"Weather tool not configured: missing OPENWEATHER_API_KEY."}

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q":city,"appid":api_key,"units":"metric"}
        response = requests.get(url,params=params,timeout=10)

        if response.status_code == 404:
            return{"error":"city not found :("}

        response.raise_for_status()
        data = response.json()
        return {
            "city":city,
            "temperature":data["main"]["temp"],
            "feels like":data["main"]["feels_like"],
            "condition":data["weather"][0]["description"]
        }
    except Exception as e:
        return{"error":str(e)}

# tool 4 - current datetime
@tool
def get_current_datetime() -> dict:
    """Get the current date and time. Never guess dates from memory."""
    return {"current datetime":datetime.now().strftime("%A, %d %B %Y, %I:%M %p")}

# tool 5 - Python REPL (py calculations)
python_repl =PythonREPLTool()

@tool("python")
def python_repl_tool(code:str) -> str:
    """Execute Python code for calculations, data analysis, and programming
    tasks. Never use this for live information (weather, prices, news, time)
    — use the dedicated tool instead. NOTE: this tool requires human approval
    before it runs (see HITL gate)."""
    return python_repl.run(code)

# tool 6 - wikipedia
@tool
def wikipedia_tool(query:str) -> str:
    """Search Wikipedia for established, encyclopedic facts."""
    try:
        wiki = WikipediaAPIWrapper()
        result = wiki.run(query)
        return result.strip() if result and result.strip() else f"o Wikipedia results found for '{query}'."
    except Exception as e:
        return f"Wikipedia search failed (try rephrasing the query): {e}"

# tool 7 - exchange rates (e.g USD -> INR)
@tool
def get_exchange_rate(from_currency:str, to_currency:str) -> str:
    """Get the LIVE exchange rate between two currencies."""
    try:
        url = "https://api.frankfurter.app/latest"
        params = {"from":from_currency.upper(),"to":to_currency.upper()}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency.upper()]
        return f"1 {from_currency.upper()} = {rate} {to_currency.upper()} (as of {data['date']})"
    except Exception as e:
        return f"Exchange rate error: {e}"

# tool 8 - RAG
from backend.rag import search_study_notes

tools = [
    search_tool,
    wikipedia_tool,
    python_repl_tool,
    get_stock_price,
    get_current_datetime,
    get_weather,
    get_exchange_rate,
    search_study_notes
]

# CONCEPT: HITL. Tools listed here pause for human approval before executing
RISKY_TOOLS = {"python"}
