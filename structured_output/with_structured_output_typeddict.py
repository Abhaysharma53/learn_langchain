from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class review(TypedDict):
    review : Annotated[str, "Summary of the product review"]
    sentiment: Annotated[str, "Sentiment of the review, could be positive, negative or neutral"]

structured_model = model.with_structured_output(review)

result = structured_model.invoke("""This product completely failed to meet my expectations—cheap quality and poor performance.
Wouldn’t recommend it to anyone; total waste of money.""")

print(result)