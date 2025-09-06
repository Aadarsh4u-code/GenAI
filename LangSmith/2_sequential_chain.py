from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = "Sequential LLM App"

hugging_face_llm = HuggingFaceEndpoint(
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="conversational", 
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    temperature=0.5
)
model = ChatHuggingFace(llm=hugging_face_llm)


prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
config = {
    'run_name': 'sequential_chain',
    'tags': ['llm app', 'report generation', 'summarization'],
    'metadata': {'model': 'meta-llama/Meta-Llama-3-8B-Instruct', 'model_temperature': 0.5, 'parser': 'StrOutputParser'}
}

result = chain.invoke({'topic': 'Unemployment in Nepal'}, config=config)

print(result)
