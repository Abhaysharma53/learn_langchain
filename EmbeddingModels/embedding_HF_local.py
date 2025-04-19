from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

docs = [
    "Delhi is the capital of India",
    "Hyderabad is the capital of Telangana",
    "Amaravathi is the capital of Andhra Pradesh",
    "Paris is the capital of France"
]

result = embedding.embed_documents(docs)
print(str(result))

