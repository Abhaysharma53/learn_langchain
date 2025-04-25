from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI()

#template 1
template1 = PromptTemplate(template= "Create a detailed summary on the {topic}", 
                           input_variables=['topic'])

#template2
template2 = PromptTemplate(template= "Create a 2 line summary of the {text}",
                           input_variables= ['text'])


parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'Langsmith in generative AI'})

print(result)