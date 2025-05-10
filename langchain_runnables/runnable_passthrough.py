from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
Prompt1 = PromptTemplate(template= "Create joke on the {topic}",
                         input_variables= ['topic'])

prompt2 = PromptTemplate(template= "Explain the joke - {text}")
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(Prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({'topic': 'Deep Learning'})
print(result)