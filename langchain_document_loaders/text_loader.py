from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template = 'Explain this poem of cricket \n {poem}',
                        input_variables= ['poem'])

loader = TextLoader('cricket.txt', encoding= 'utf-8')

docs = loader.load()
parser = StrOutputParser()
#print(docs[0].metadata)
#print(docs[0].page_content)

chain = prompt | model | parser

result = chain.invoke({'poem': docs[0].page_content})

print(result)