from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

template1 = PromptTemplate(template= "Give a detailed summary report on the {topic}",
                           input_variables= ['topic'])

template2 = PromptTemplate(template= "Create a 5 pointer summary out of the {text}",
                           input_variables= ['text']
                        )

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'recurrent neural network'})
print(result)