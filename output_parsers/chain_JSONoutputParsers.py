from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

#create Json Output parser
parser = JsonOutputParser()


#template2
template = PromptTemplate(template= "Give me 5 points about the {topic} \n {format_instruction}",
                           input_variables= ['topic'],
                           partial_variables= {'format_instruction': parser.get_format_instructions()})

chain = template | model | parser

result = chain.invoke({'topic': 'Agents in Generative AI'})
print(result)

# final_output = parser.parse(result.content)
# print(final_output)