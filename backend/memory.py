import uuid
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from backend.config import LLM
from backend.state import ChatState

CONTEXT_RETRIEVAL_LIMIT = 3
DEDUP_RETRIEVAL_LIMIT = 8


def _get_latest_message(state:ChatState) -> HumanMessage | None:
    return next((m for m in reversed(state["messages"]) if isinstance(m,HumanMessage)),None)

def _memory_namespace(config:RunnableConfig) -> tuple:
    user_id = config["configurable"].get("user_id","default_user")
    return (user_id,"memories")

# structured output schema for memory extraction
class MemoryItem(BaseModel): # to tell if memory is unique or duplicated.
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(
        description=(
            "True if this memory adds genuinely new information. "
            "False if it is duplicate or semantically equivalent "
            "to an existing memory."
        )
    )

class MemoryDecision(BaseModel):
    should_wirte: bool = Field(description="Whether any new long-term memory should be stored")
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description="Atomic user memories to consider storing",
    )

# extractor llm
memory_extractor = LLM.with_structured_output(MemoryDecision)

# memory prompt telling LLM to save memory
MEMORY_PROMPT = """You are responsible for maintaining accurate long-term user memory.

CURRENT RELEVANT USER MEMORIES:
{existing_memory}

TASK:
- Review the user's latest message.
- Extract only stable, user-specific information worth remembering long-term.
- Useful categories include identity, stable preferences, ongoing projects, goals,
  technical stack, learning interests, and other durable context.
- Do NOT store transient information such as greetings, one-time tasks, or the
  current conversation topic unless it represents stable user information.
- For each extracted memory, set is_new=true ONLY if it adds information that
  is not already present in the existing memories.
- Treat semantically equivalent statements as duplicates even if the wording differs.
- If the user corrects an existing fact, prefer the corrected fact rather than
  blindly creating another contradictory memory.
- Keep every memory short and atomic.
- Do not speculate. Only use facts explicitly stated by the user.
- If nothing is worth storing, return should_write=false and an empty list.
"""

def load_memory_node(state:ChatState,config:RunnableConfig,store:BaseStore):
    """Semantic retrieval of memories relevant to the user's latest message,
    so chat_node (backend/agent.py) can personalize its reply."""

    namespace = _memory_namespace(config)

    last_message = _get_latest_message(state)
    if last_message is None:
        return{"retrieved_memories":""}

    hits = store.search(namespace,query=last_message.content,limit=CONTEXT_RETRIEVAL_LIMIT)

    if not hits:
        return {"retrieved_memories":""}
    return {"retrieved_memories":"\n".join(f"- {h.value["content"]}" for h in hits)}

def save_memory_node(state:ChatState,config:RunnableConfig,store:BaseStore):
    """Extract candidate facts from the user's latest message, compare them
    against existing memories, and write only the ones the LLM judges to be
    genuinely new — this is the semantic-duplicate-detection step."""

    namespace = _memory_namespace(config)

    last_message = _get_latest_message(state)
    if last_message is None:
        return {}

    relevant_items = store.search(namespace,query=last_message.content, limit=DEDUP_RETRIEVAL_LIMIT)

    if relevant_items:
        existing_memory = "\n".join(
            f"- {item.value.get("content","")}" for item in relevant_items if item.value.get("content") 
        )
    else:
        existing_memory = "(empty)"

    try:
        decision: MemoryDecision = memory_extractor.invoke(
            [
                SystemMessage(content=MEMORY_PROMPT.format(existing_memory=existing_memory)),
                last_message,
            ]
        )
    except Exception as e:
        print(f"save_memory_node error (skipping memory save this turn): {e}")
        return {}

    if decision.should_wirte:
        for mem in decision.memories:
            if mem.is_new:
                store.put(namespace,str(uuid.uuid4()),{"content":mem.text})

    return {}