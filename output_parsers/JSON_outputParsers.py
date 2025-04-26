from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

#create Json Output parser
parser = JsonOutputParser()

# #template1
# template1 = PromptTemplate(template= "give me a detailed summary report on {topic}",
#                            input_variables= ['topic'],
#                            partial_variables= {'format_instruction': parser.get_format_instructions()})


#template2
template = PromptTemplate(template= "Give me 5 points about the {topic} \n {format_instruction}",
                           input_variables= ['topic'],
                           partial_variables= {'format_instruction': parser.get_format_instructions()})

prompt = template.format(topic = 'Agents in Generative AI')
#print(prompt)

result = model.invoke(prompt)
#print(result)
final_output = parser.parse(result.content)
print(final_output)