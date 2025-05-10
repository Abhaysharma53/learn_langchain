from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(template= "Create a detailed report on the {topic}",
                         input_variables= ['topic'])

prompt2 = PromptTemplate(template= "Create a summary out of the {text}",
                         input_variables= ['text'])
parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

condition_chain = RunnableBranch(
    (lambda x: len(x) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough
)

final_chain = RunnableSequence(report_gen_chain, condition_chain)
result = final_chain.invoke({'topic': 'MCP in agentic AI'})
print(result)