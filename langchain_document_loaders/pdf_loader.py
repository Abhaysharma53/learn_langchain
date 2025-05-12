from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
parsers = StrOutputParser()

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)