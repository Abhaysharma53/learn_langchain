from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import  RunnableSequence
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()

prompt = PromptTemplate(template= "Create a joke on the {topic}",
                        input_variables= ['topic'])

prompt2 = PromptTemplate(template= 'Explain the folowing joke - {text}',
                         input_variables= ['text'])

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)
print(chain.invoke({'topic':'LLM'}))