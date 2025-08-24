from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()       

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, api_key=os.getenv("OPENAI_API_KEY"))   

messages = [
    SystemMessage(content="You are a helpful research assistant."),
    HumanMessage(content="Explain the concept of attention mechanism in neural networks.")      
]

response = chat_model.invoke(messages)
messages.append(AIMessage(content=response.content))
print("Messages:", messages)  
