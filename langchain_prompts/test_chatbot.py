from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()
chat_history = list(SystemMessage(
    content="You are a helpful assistant who answers the user's questions patiently and accurately. "
))
while True:
    user_input = input("User : ")
    chat_history.append(HumanMessage(content= user_input)) 
    if user_input.lower() == "exit":
        break
    else:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content= result.content))
        print(f"Assistant : {result.content}")
 
print(chat_history)