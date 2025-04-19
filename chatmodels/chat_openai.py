from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4', temperature = 2)
#temprature = 0.7 is default, 2 is max, 0 is min -its a randomness factor or creativity factor

result = model.invoke("give me a flirty pickup line to impress my crush", max_tokens = 20)
#print(result) #alot of other info is also avaial

print(result.content)