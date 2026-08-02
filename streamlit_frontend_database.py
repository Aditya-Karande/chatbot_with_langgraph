import streamlit as st
from langgraph_backend_database import (
    chatbot,
    generate_title,
    save_thread,
    get_all_threads,
    rename_title,
    delete_thread
)
from langchain_core.messages import HumanMessage
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
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

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

            loaded_messages = []

            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                loaded_messages.append({"role":role, "content":msg.content})

            st.session_state['message_history'] = loaded_messages

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

    CONFIG = {"configurable":{"thread_id":st.session_state['thread_id']}}

    # Streaming.
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages":HumanMessage(content=user_input)},
                config=CONFIG,
                stream_mode="messages"
            )
        )

    # add message to message_history
    st.session_state["message_history"].append({"role":"assistant", "content":ai_message})
    st.rerun()