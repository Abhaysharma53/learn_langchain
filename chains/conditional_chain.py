from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser

load_dotenv()


model = ChatOpenAI()
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description= "the overall sentiment of the review")

parser2 = PydanticOutputParser(pydantic_object= Feedback)
template1 = PromptTemplate(template= "Based upon the review identify the sentiment and classifiy the sentiment either as positive or negative \n {review} \n {format_instruction}",
                          input_variables= ['review'],
                          partial_variables= {'format_instruction': parser2.get_format_instructions()})

template2 = PromptTemplate(template= "The customer had a great experience with our products let's try to dig deep and get their feedback on what they liked the most \n {review}",
                           input_variables= ['review'])

template3 = PromptTemplate(template= "The customer had a bad experience with our products let's try to symphatesize with them and understand what went wrong with a assured commitment to make sure this won't happen again \n {review}",
                           input_variables= ['review'])

classifier_chain = template1 | model | parser2

review = """I had high hopes for this laptop, but it has been a major letdown from day one. The system is painfully slow, even with basic tasks like browsing and opening simple documents. The battery life is a joke — barely lasts 2-3 hours even when fully charged. On top of that, the laptop heats up quickly and the fan noise is loud enough to be distracting during meetings.

Build quality also feels cheap. The keyboard is flimsy, and the screen brightness is uneven with noticeable color distortion. Customer support was no help either; they kept pushing firmware updates that did not solve any of the core issues. For the price I paid, I expected much better performance and reliability. Definitely regret this purchase and would not recommend it to anyone looking for a dependable machine."""


review2 = """I've been using this laptop for a few weeks now, and I couldn't be happier with my decision. The performance is incredibly smooth — multitasking between heavy applications is seamless, and boot-up times are impressively fast. The battery easily lasts me a full workday, which has been a huge bonus for my on-the-go schedule.

The build quality feels premium, with a sturdy chassis and a keyboard that's comfortable to type on for long hours. The display is vibrant and sharp, making streaming and designing a real pleasure. I was also impressed by how quietly and efficiently it runs, even during demanding tasks.

Overall, this laptop strikes a perfect balance between power, portability, and style. Highly recommended for both professionals and students looking for a reliable machine!

"""
# result = chain.invoke({'review': review}).sentiment
#print(result)/

"""
RunnableBranch(
(condition1, chain1),
(condition2, chain2),
default chain
)
"""
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', template2 | model | parser),
    (lambda x: x.sentiment == 'Negative', template3 | model | parser),
    RunnableLambda(lambda x : "Could not find sentiment")
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({'review': review2})
print(result)