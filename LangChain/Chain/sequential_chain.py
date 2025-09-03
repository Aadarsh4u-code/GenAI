import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser          

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

# Define the model
chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9, api_key=os.getenv('OPENAI_API_KEY'))

parser = StrOutputParser()

chain = prompt1 | chat_model | parser | prompt2 | chat_model | parser

result = chain.invoke({'topic':'Future of Data Scientist in next 10 years'})
print(result)

chain.get_graph().print_ascii()