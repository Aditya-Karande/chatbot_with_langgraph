import streamlit as st
from langgraph_backend_tools import (
    chatbot,
    generate_title,
    save_thread,
    get_all_threads,
    rename_title,
    delete_thread
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================================== Utility Function ===========================================================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['thread_saved'] = False # not in db yet - no title until first msg..

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable":{"thread_id":thread_id}})
    return state.values.get('messages', [])

def messages_to_display(messages):
    """Convert LangGraph's raw message list into clean {role, content} dicts
    for the UI — only real Human/AI text, never ToolMessages (raw JSON) or
    AIMessages that only carried a tool_call with no visible content."""
    display = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            display.append({"role": "assistant", "content": msg.content})
        # ToolMessage and empty tool-call AIMessages are intentionally skipped
    return display

# =========================================== Session Setup ===========================================================
# st.session_state -> dict
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    reset_chat()

if 'thread_saved' not in st.session_state:
    st.session_state['thread_saved'] = False

# source of truth for the sidebar list is now the DB, not session_state
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'editing_thread' not in st.session_state:
    st.session_state['editing_thread'] = None # thread_id currently being renamed..

# =========================================== Sidebar UI ===========================================================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread in st.session_state['chat_threads']:

    tid = thread['thread_id']
    title = thread['title']

    col1, col2, col3 = st.sidebar.columns([8,3,3])

    if st.session_state['editing_thread'] == tid:
        new_title = col1.text_input(
            "rename", value=title, key= f"edit_{tid}",label_visibility="collapsed"
        )

        if col2.button("✅", key=f"save_{tid}"):
            rename_title(tid, new_title)
            st.session_state['chat_threads'] = get_all_threads()
            st.session_state['editing_thread'] = None
            st.rerun()
    else:
        if col1.button(title, key=f"open_{tid}"):
            st.session_state['thread_id'] = tid
            st.session_state['thread_saved'] = True
            messages = load_conversation(tid)
            st.session_state['message_history'] = messages_to_display(messages)

        if col2.button("✏️", key=f"rename_{tid}"):
            st.session_state['editing_thread'] = tid
            st.rerun()

        if col3.button("🗑️", key=f"delete_{tid}"):
            delete_thread(tid)
            st.session_state['chat_threads'] = get_all_threads()
            if st.session_state['thread_id'] == tid:
                reset_chat()
            st.rerun()

# =========================================== Main UI ===========================================================
# loading the conversation history
for messages in st.session_state['message_history']:
    with st.chat_message(messages["role"]):
        st.text(messages["content"])

user_input = st.chat_input('Type Here..')

if user_input:

    # add message to message_history
    st.session_state["message_history"].append({"role":"user", "content":user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # first message in brand-new thread -> generate + save title
    if not st.session_state['thread_saved']:
        title = generate_title(user_input)
        save_thread(st.session_state['thread_id'], title)
        st.session_state['thread_saved'] = True
        st.session_state['chat_threads'] = get_all_threads()

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Streaming.
    with st.chat_message('assistant'):
        # use a mutable holder so that generator can set/modify it.
        status_holder = {"box":None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages":HumanMessage(content=user_input)},
                config=CONFIG,
                stream_mode="messages"
            ):
                # lazily create and update the SAME status container when any tool runs..
                if isinstance(message_chunk,ToolMessage):
                    tool_name = getattr(message_chunk,"name","tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label = f"🔧 using `{tool_name}` ...",
                            state = "running",
                            expanded= True
                        )

                # to disply only AI messages and not tool calls...
                if isinstance(message_chunk, AIMessage):
                    if message_chunk.content:
                        yield message_chunk.content  


                print(message_chunk) 

        ai_message = st.write_stream(ai_only_stream())

        # finalize only if a tool was actually used.
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # add message to message_history
    st.session_state["message_history"].append({"role":"assistant", "content":ai_message})
    st.rerun()