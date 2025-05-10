from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(template= "Create a tweet on {topic}",
                         input_variables= ['topic'])

prompt2 = PromptTemplate(template= "Create a linkedin post about {topic}",
                         input_variables= ['topic'])

model = ChatOpenAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin_post': RunnableSequence(prompt2, model, parser)
})

#print(parallel_chain.invoke({'topic':'India Pakistan relations'}))

result = parallel_chain.invoke({'topic':'AI Agents'})
print(result['linkedin_post'])