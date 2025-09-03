import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import  StrOutputParser
from langchain.schema.runnable import RunnableParallel         

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate a short and simple note on \n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 short question and answer from this note \n {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document notes -> \n {notes} and quiz -> \n {quiz}",
    input_variables=["notes", "quiz"]
)

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

parser = StrOutputParser()

# Define the first model from HuggingFace
chat_model1 = ChatHuggingFace(llm=llm)

# Define the second model from OpenAI
chat_model2 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9, api_key=os.getenv('OPENAI_API_KEY'))

# Create RunnableParallel chain
parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | chat_model1 | parser,
        "quiz": prompt2 | chat_model2 | parser
    }
)

# Now create a  merge chain
merge_chain = prompt3 | chat_model2 | parser

# FInal chain that first runs the parallel chain and then the merge chain
final_chain = parallel_chain | merge_chain

text = "Linear Regression"

result = final_chain.invoke({"topic": text})
print(result)

final_chain.get_graph().print_ascii()