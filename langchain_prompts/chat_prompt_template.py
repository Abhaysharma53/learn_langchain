from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatOpenAI()

chat_template = ChatPromptTemplate([
    ('system', "you are a helpful {domain} expert"),
    ('human', "explain in simple terms, what is {topic}")

])

prompt = chat_template.invoke({
    "domain": "AI",
    "topic": "Generative AI"
})
print(prompt)