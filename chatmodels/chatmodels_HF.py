from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
  repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  task = "text-generation"
 )

model = ChatHuggingFace(llm = llm, temperature = 1.5)
result = model.invoke("What is the capital of andhra pradesh", max_tokens = 25)
print(result.content)