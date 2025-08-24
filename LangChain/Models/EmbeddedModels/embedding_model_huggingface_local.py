from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", dimensions=20)


# For Multiple Queries
# result = embeddings.embed_documents(["Hello, world!", "Goodbye!"])

# For single query
result = embeddings.embed_query("Hello, world!")

print(result)