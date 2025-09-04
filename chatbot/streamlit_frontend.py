import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

st.title("💬 Custom Chat UI")


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

CONFIG = {'configurable': {'thread_id': 'thread-1'}}


# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type Here ...")

if user_input:

    # First add the message to chat_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    # Get AI message
    
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    print(response)
    ai_message = response['messages'][-1].content
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)







# messages = [
#     {"sender": "ai", "message": "Hello! How can I help you today?"},
#     {"sender": "user", "message": "Tell me about LangGraph."},
#     {"sender": "ai", "message": "LangGraph is a framework for stateful AI apps."},
# ]

# for msg in messages:
#     if msg["sender"] == "user":
#         col1, col2 = st.columns([2, 1])
#         with col2:
#             st.markdown(
#                 f"<div style='background-color:#2563eb;color:white;padding:8px;border-radius:12px;text-align:right;'>{msg['message']}</div>",
#                 unsafe_allow_html=True,
#             )
#     else:
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st.markdown(
#                 f"<div style='background-color:#e5e7eb;color:black;padding:8px;border-radius:12px;text-align:left;'>{msg['message']}</div>",
#                 unsafe_allow_html=True,
#             )
