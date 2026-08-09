import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from backend import (
    chatbot,
    delete_thread,
    generate_title,
    get_all_threads,
    ingest_files,
    rename_title,
    save_thread,
)

NOTES_DIR = "./notes"

# =========================================== Utility Function ===========================================================


def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["thread_saved"] = False


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def messages_to_display(messages):
    display = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            display.append({"role": "assistant", "content": msg.content})
    return display


def get_pending_interrupt(thread_id):
    """Checks whether the graph is currently paused mid-run waiting for
    human approval (i.e. hitl_gate called interrupt() and hasn't been
    resumed yet). Returns the interrupt payload, or None."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def stream_turn(payload, config):
    """Shared streaming logic for both a fresh user message and a resumed
    Command after HITL approval/rejection. Returns the final assistant text
    (empty string if the run paused on an interrupt instead of finishing)."""
    status_holder = {"box": None}

    def ai_only_stream():
        for message_chunk, metadata in chatbot.stream(
            payload, config=config, stream_mode="messages"
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_holder["box"] is None:
                    status_holder["box"] = st.status(f"🔧 using `{tool_name}` ...", expanded=True)
                else:
                    status_holder["box"].update(label=f"🔧 using `{tool_name}` ...", state="running", expanded=True)

            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                yield message_chunk.content

    ai_message = st.write_stream(ai_only_stream())

    if status_holder["box"] is not None:
        status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    return ai_message or ""


# =========================================== Session Setup ===========================================================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    reset_chat()

if "thread_saved" not in st.session_state:
    st.session_state["thread_saved"] = False

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "editing_thread" not in st.session_state:
    st.session_state["editing_thread"] = None

if "user_id" not in st.session_state:
    # scopes long-term memory — swap for real auth later, this is enough to
    # demo cross-thread recall for one person
    st.session_state["user_id"] = "aditya"

# =========================================== Sidebar UI ===========================================================
st.sidebar.title("LangGraph Chatbot")

st.session_state["user_id"] = st.sidebar.text_input("User ID (for long-term memory)", value=st.session_state["user_id"])

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("📚 Study Notes (RAG)")
uploaded_files = st.sidebar.file_uploader(
    "Upload notes for the chatbot to search",
    type=["pdf", "md", "txt"],
    accept_multiple_files=True,
)
if uploaded_files and st.sidebar.button("Ingest into RAG"):
    os.makedirs(NOTES_DIR, exist_ok=True)
    saved_paths = []
    for f in uploaded_files:
        path = os.path.join(NOTES_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        saved_paths.append(path)
    with st.spinner("Embedding and storing your notes..."):
        count = ingest_files(saved_paths)
    st.sidebar.success(f"Ingested {count} chunks from {len(saved_paths)} file(s).")

st.sidebar.header("My Conversations")

for thread in st.session_state["chat_threads"]:
    tid = thread["thread_id"]
    title = thread["title"]

    col1, col2, col3 = st.sidebar.columns([8, 3, 3])

    if st.session_state["editing_thread"] == tid:
        new_title = col1.text_input("rename", value=title, key=f"edit_{tid}", label_visibility="collapsed")
        if col2.button("✅", key=f"save_{tid}"):
            rename_title(tid, new_title)
            st.session_state["chat_threads"] = get_all_threads()
            st.session_state["editing_thread"] = None
            st.rerun()
    else:
        if col1.button(title, key=f"open_{tid}"):
            st.session_state["thread_id"] = tid
            st.session_state["thread_saved"] = True
            messages = load_conversation(tid)
            st.session_state["message_history"] = messages_to_display(messages)

        if col2.button("✏️", key=f"rename_{tid}"):
            st.session_state["editing_thread"] = tid
            st.rerun()

        if col3.button("🗑️", key=f"delete_{tid}"):
            delete_thread(tid)
            st.session_state["chat_threads"] = get_all_threads()
            if st.session_state["thread_id"] == tid:
                reset_chat()
            st.rerun()

# =========================================== Main UI ===========================================================

for messages in st.session_state["message_history"]:
    with st.chat_message(messages["role"]):
        st.text(messages["content"])

CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"],
        "user_id": st.session_state["user_id"],
    },
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_turn",
}

# --- HITL: if the graph is paused waiting for approval, show that instead
# of the normal chat input. This survives page reloads because the pause
# lives in Postgres checkpointer state, not in Streamlit session state.
pending = get_pending_interrupt(st.session_state["thread_id"])

if pending:
    st.warning(pending["message"])
    for call in pending["risky_calls"]:
        st.code(f"{call['name']}({call['args']})", language="python")

    col1, col2 = st.columns(2)
    with st.chat_message("assistant"):
        if col1.button("✅ Approve", use_container_width=True):
            ai_message = stream_turn(Command(resume={"approved": True}), CONFIG)
            if ai_message:
                st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            st.rerun()
        if col2.button("🚫 Reject", use_container_width=True):
            ai_message = stream_turn(Command(resume={"approved": False}), CONFIG)
            if ai_message:
                st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
            st.rerun()

else:
    user_input = st.chat_input("Type Here..")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        if not st.session_state["thread_saved"]:
            title = generate_title(user_input)
            save_thread(st.session_state["thread_id"], title)
            st.session_state["thread_saved"] = True
            st.session_state["chat_threads"] = get_all_threads()

        with st.chat_message("assistant"):
            ai_message = stream_turn({"messages": [HumanMessage(content=user_input)]}, CONFIG)

        # ai_message will be empty if the run paused on hitl_gate instead of
        # finishing — in that case we DON'T append a fake assistant turn,
        # we just rerun so the interrupt UI above takes over.
        if ai_message:
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        st.rerun()
