from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
import os

from dotenv import load_dotenv


load_dotenv()

hugging_face_llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="conversational", 
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03
)
llm = ChatHuggingFace(llm=hugging_face_llm)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Create sqlite database
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

# Test
# CONFIG = {'configurable': {'thread_id':'thread-1'}}
# response = chatbot.invoke(
#             {'messages': [HumanMessage(content='what is my name')]}, 
#             config=CONFIG,
#         )

# print("response",response)

