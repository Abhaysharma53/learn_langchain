from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAI(model = 'gpt-3.5-turbo-instruct')
result = llm.invoke("my name is Abhay and i work as data scientist. what are the must skills for a data scientist these days to switch to a better job?")

print(result)