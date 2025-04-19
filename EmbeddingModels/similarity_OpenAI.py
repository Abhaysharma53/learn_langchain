from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embedding = OpenAIEmbeddings(model = 'text-embedding-3-small', dimensions = 312)

docs = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query_string = "Who is the 'God of Cricket'?"
documents = embedding.embed_documents(docs)

query = embedding.embed_query(query_string)
similarity = cosine_similarity([query], documents)[0]
#print(list(enumerate(similarity)))
index, score = sorted(list(enumerate(similarity)), key = lambda x: x[1], reverse= True)[0]
print(query_string)
print(docs[index], score)
