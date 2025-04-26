from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()
model = ChatOpenAI()

schema = [
    ResponseSchema(name= 'fact_1', description= 'fact_1 about the topic'),
    ResponseSchema(name= 'fact_2', description= 'fact_2 about the topic'),
    ResponseSchema(name= 'fact_3', description= 'fact_3 about the topic')
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(template = "GIve 3 intresting pointers about the {topic} \n {format_instruction}",
                          input_variables= ['topic'],
                          partial_variables= {'format_instruction': parser.get_format_instructions()})

# prompt = template.format(topic = "Agentic RAG")
# print(prompt)
chain = template | model | parser
result = chain.invoke({'topic':'AI Agents in Generative AI'})
print(result)
