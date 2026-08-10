from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agent import chat_node, hitl_gate, route_after_agent
from backend.config import checkpointer, store
from backend.memory import load_memory_node, save_memory_node
from backend.state import ChatState
from backend.summarization import summarize_node
from backend.tools import tools

# graph
graph = StateGraph(ChatState)

graph.add_node("load_memory",load_memory_node)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",ToolNode(tools=tools))
graph.add_node("hitl_gate",hitl_gate)
graph.add_node("save_memory",save_memory_node)
graph.add_node("summarize",summarize_node)

graph.add_edge(START,"load_memory")
graph.add_edge("load_memory","chat_node")
graph.add_conditional_edges(
    "chat_node",
    route_after_agent,
    {"tools":"tools","hitl_gate":"hitl_gate","save_memory":"save_memory"},
    )
graph.add_edge("tools","chat_node")
# hitl_gate has no static outgoing edge it returns Command(goto=..) itself
graph.add_edge("save_memory","summarize")
graph.add_edge("summarize",END)

chatbot = graph.compile(checkpointer=checkpointer,store=store)