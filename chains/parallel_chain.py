from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

model1 = ChatOpenAI()
model2 = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(template = "Generate short and simple notes about the machine learning algorithm from the following text \n {text}",
                           input_variables= ['text'])

prompt2 = PromptTemplate(template = "Generate five short question about the machine learning algorithm based upon the text \n {text}",
                           input_variables=['text'])

prompt3 = PromptTemplate(template = 'Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
                         input_variables= ['notes', 'quiz'])

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser
final_chain = parallel_chain | merge_chain
text = """The k-nearest neighbors (KNN) algorithm is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point. It is one of the popular and simplest classification and regression classifiers used in machine learning today.

While the KNN algorithm can be used for either regression or classification problems, it is typically used as a classification algorithm, working off the assumption that similar points can be found near one another.

For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature. The distinction between these terminologies is that “majority voting” technically requires a majority of greater than 50%, which primarily works when there are only two categories. When you have multiple classes—e.g. four categories, you don’t necessarily need 50% of the vote to make a conclusion about a class; you could assign a class label with a vote of greater than 25%.
"""

result = final_chain.invoke({'text':text})
print(result)