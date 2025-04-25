from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatOpenAI()

chat_template = ChatPromptTemplate([
    ('system', "you are a helpful customer service agent"),
    MessagesPlaceholder(variable_name= 'chat_history'),
    ('human', "{query}")
])
chat_history = list()
with open('chat_history.txt', 'r') as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({'chat_history':chat_history, 'query': 'Where is my refund'})

print(prompt)