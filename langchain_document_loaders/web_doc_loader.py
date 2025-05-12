from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()
prompt = PromptTemplate(template= "Based upon the following {text} \n - answer the following question as per your best capability-  {question}",
                        input_variables= ['text', 'question'])

parser = StrOutputParser()

url = "https://www.flipkart.com/nothing-phone-3a-white-128-gb/p/itm49557c5a65f9c"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text': docs[0].page_content, 'question': "What are the qualities that stood out for the product we are talking about here"})
print(result)