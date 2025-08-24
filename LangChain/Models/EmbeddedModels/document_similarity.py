import os
from langchain_openai import OpenAIEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    api_key=os.getenv("OPENAI_API_KEY"),
    dimensions=300
    )

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli."

# query = "Who is bowler of India?"

documents_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)
similarities = cosine_similarity([query_embedding], documents_embeddings)
score = np.max(similarities)
index = np.argmax(similarities)
# index, score = sorted(list(enumerate(similarities)), key=lambda x: x[1])[-1]

print(query)
print(f"Most similar document is: '{documents[index]}' with similarity score of {score}")
