import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, api_key=os.getenv("OPENAI_API_KEY"))

result = chat_model.invoke("What would be a good company name for a company that makes colorful socks?")
print(result)