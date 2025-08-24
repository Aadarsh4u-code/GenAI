from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=10
    )

# For Multiple Queries
result = embeddings.embed_documents(["Hello, world!", "Goodbye, world!"])

# For single query
# result = embeddings.embed_query("Hello, world!")

print(result)