import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import os

load_dotenv()

model = ChatOpenAI()



#st.title("LLM research assistant")
st.header("Research Assistant")
paper_input = st.selectbox("Select a paper", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select a style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select a length", ["Short (1-2 Paragraph)", "Medium (3-5 Paragraph)", "Long (Detailed Explanation)"])
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""Please summarize the research paper titled \"{paper_input}\" with the following specifications:
    Explanation Style: {style_input}  
    Explanation Length: {length_input}  
    1. Mathematical Details:  
       - Include relevant mathematical equations if present in the paper.  
       - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
       2. Analogies:  
       - Use relatable analogies to simplify complex ideas.  
       If certain information is not available in the paper, respond with: \"Insufficient information available\" instead of guessing.  
       Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
       validate_template= True)

prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

result = model.invoke(prompt)
if st.button("Summarize"):
    st.write(result.content)


