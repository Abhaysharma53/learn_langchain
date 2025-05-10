from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatOpenAI()
prompt = PromptTemplate(template= "Create a joke on the {topic}",
                        input_variables= ['topic'])

parser = StrOutputParser()

joke_generate = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_generate, parallel_chain)

result = final_chain.invoke({'topic': 'Data Scientist'})
#print(result)

final_result = """{} \n word count - {}""".format(result['joke'], result['length'])
print(final_result)