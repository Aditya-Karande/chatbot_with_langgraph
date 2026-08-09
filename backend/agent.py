# agent loop + HITL

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt

from backend.config import LLM
from backend.state import SYSTEM_PROMPT_BASE, ChatState
from backend.tools import RISKY_TOOLS, tools

llm_with_tools = LLM.bind_tools(tools)

def chat_node(state: ChatState) -> dict:
    system_content = SYSTEM_PROMPT_BASE
    if state.get("summary"):
        system_content += f"\n\nSummary of earlier conversation:\n{state['summary']}"
    if state.get("retrieved_memories"):
        system_content += f"\n\nThings you remember about this user:\n{state['retrieved_memories']}"

    full_messages = [SystemMessage(content=system_content)] + state["messages"]
    try:
        response = llm_with_tools.invoke(full_messages)
    except Exception as e:
        print(f"chat_node error: {e}")
        response = AIMessage(content=f"Something went wrong on my end: {e}")
    return {"messages": [response]}

def route_after_agent(state:ChatState) -> Literal["tools","hitl_gate","save_memory"]:

    last = state["messages"][-1]
    calls = getattr(last,"tool_calls", None)

    if not calls:
        return "save_memory"
    if any(tc["name"] in RISKY_TOOLS for tc in calls):
        return "hitl_gate"
    return "tools"

def hitl_gate(state:ChatState) -> Command[Literal["tools","chat_node"]]:
    """Pauses the graph before a risky tool runs. `interrupt()` stops
    execution here and returns its payload to whoever is watching the run
    (the Streamlit frontend). Resuming happens via
    chatbot.stream(Command(resume=...), config=...)."""
    last = state["messages"][-1]
    all_calls = last.tool_calls or []
    risky_calls = [tc for tc in all_calls if tc["name"] in RISKY_TOOLS]

    decision = interrupt(
        {
            "message":"The agent wants to run a tool that can execute code. Approve?",
            "risky_calls":[{"name":tc["name"],"args":tc["args"]} for tc in risky_calls],
        }
    )

    if decision.get("approved"):
        return Command(goto="tools")

    rejection_msgs = [
        ToolMessage(content="User did not approve this action.",tool_call_id = tc["id"]) for tc in all_calls
    ]

    return Command(goto="chat_node",update={"messages":rejection_msgs})
