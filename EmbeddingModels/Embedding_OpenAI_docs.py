from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 32)

docs = [
    "Delhi is the capital of India",
    "Hyderabad is the capital of Telangana",
    "Amaravathi is the capital of Andhra Pradesh",
    "Paris is the capital of France",
]
result = embedding.embed_documents(docs)
print(str(result))