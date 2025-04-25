from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class review(TypedDict):
    review : str
    sentiment: str

structured_model = model.with_structured_output(review)

result = structured_model.invoke("""This product completely failed to meet my expectations—cheap quality and poor performance.
Wouldn’t recommend it to anyone; total waste of money.""")

print(result)