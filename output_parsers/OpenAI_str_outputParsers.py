from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI()

#template 1
template1 = PromptTemplate(template= "Create a detailed summary on the {topic}", 
                           input_variables=['topic'])

#template2
template2 = PromptTemplate(template= "Create a 2 line summary of the {text}",
                           input_variables= ['text'])

prompt1 = template1.invoke({'topic': 'Generative AI'})
#print(prompt1)
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result.content})
final_result = model.invoke(prompt2)

print(final_result.content)