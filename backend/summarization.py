# short term memory

from langchain_core.messages import HumanMessage,RemoveMessage

from backend.config import LLM
from backend.state import ChatState

SUMMARY_TRIGGER = 12 # once the thread exceeds this many messages, condense

def summarize_node(state:ChatState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARY_TRIGGER:
        return {}

    keep_last = 4
    old_messages = messages[:-keep_last]
    existing_summary = state.get("summmary","")

    transcript = "\n".join(f"{m.type}:{m.content}" for m in old_messages if m.content)

    new_summary = LLM.invoke(
        [
            HumanMessage(
                content=(
                    "Extend the running summary with these new messages. "
                    "Keep only facts and decisions relevant to future turns. "
                    "Be concise.\n\n"
                    f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
                    f"New messages:\n{transcript}"
                )
            )
        ]
    ).content

    # delete old messages
    delete_ops = [RemoveMessage(id=m.id) for m in old_messages]

    return {"summary":new_summary, "messages":delete_ops}