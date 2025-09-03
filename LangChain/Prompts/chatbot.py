from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 

from dotenv import load_dotenv
import os   
load_dotenv()

chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input = input("Enter your message (or type 'exit' to quit): ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break

    response = chat_model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI:", response.content)

print("Chat History:", chat_history)