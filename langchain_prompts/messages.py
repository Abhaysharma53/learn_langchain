from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatOpenAI()
Message = [
    SystemMessage(
        content="You are a helpful assistant who answers the user's questions pateintly and accurately. "
    ),
    HumanMessage(
        content="Tell me about use of langchain in generative AI."
    )
]

result = model.invoke(Message)
Message.append(AIMessage(result.content))
#print(result.content)
print(Message)
