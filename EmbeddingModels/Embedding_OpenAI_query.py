from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 64)
result = embedding.embed_query("What is the capital of andhra pradesh")
print(str(result))