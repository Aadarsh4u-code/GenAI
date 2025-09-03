import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field           

load_dotenv()


prompt = PromptTemplate(
    template="Generate 5 intresting factes about {topic}",
    input_variables=["topic"]
)

# Define the model
chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9, api_key=os.getenv('OPENAI_API_KEY'))

parser = StrOutputParser()

chain = prompt | chat_model | parser

result = chain.invoke({'topic':'LangChain chains in GenAI'})
print(result)

# chain.get_graph().print_ascii()

