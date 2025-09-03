from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv  
from typing import Literal, Optional, TypedDict, Annotated

load_dotenv()  # take environment variables from .env.

chat_model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

# Define a schema TypedDict for structured output
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes mentioned in the review"]
    summary: Annotated[str, "A concise summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "The overall sentiment of the review, either 'positive', 'negative', or 'neutral'"]
    prons: Annotated[Optional[list[str]], "List the pros mentioned in the review"]
    cons: Annotated[Optional[list[str]], "List the cons mentioned in the review"]
    name: Annotated[Optional[str], "The name of the reviewer"]


review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Aadarsh Kushwaha
"""
structured_model = chat_model.with_structured_output(Review)

response = structured_model.invoke(review)
print(response)

print(f"Summary: {response['summary']}")
print(f"Sentiment: {response['sentiment']}")
print(f"Key Themes: {response['key_themes']}")
print(f"Pros: {response['prons']}")
print(f"Cons: {response['cons']}")