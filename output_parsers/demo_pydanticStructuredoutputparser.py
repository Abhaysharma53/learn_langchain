from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
model = ChatOpenAI()

class Person(BaseModel):
    name: str = Field(description= "The name of the person")
    age: int = Field(gt = 18, description = "Age of the person")
    city: str = Field(default= "New Delhi", description= 'city the person belongs to')


parser = PydanticOutputParser(pydantic_object= Person)

template = PromptTemplate(template= "Generate the name, age and city of a fictional person belong to {location} \n {format_instruction}",
                          input_variables= ['location'],
                          partial_variables= {'format_instruction': parser.get_format_instructions()})

# prompt = template.format(location = 'Chennai')
# print(prompt)
chain = template | model | parser

result = chain.invoke({'location':'Colombo'})
print(result)