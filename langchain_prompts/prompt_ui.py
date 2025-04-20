import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, load_prompt
import os

load_dotenv()

model = ChatOpenAI()



#st.title("LLM research assistant")
st.header("Research Assistant")
paper_input = st.selectbox("Select a paper", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select a style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
length_input = st.selectbox("Select a length", ["Short (1-2 Paragraph)", "Medium (3-5 Paragraph)", "Long (Detailed Explanation)"])

template = load_prompt("research_paper_summary_template.json")

# prompt = template.invoke({
#     "paper_input": paper_input,
#     "style_input": style_input,
#     "length_input": length_input
# })

#result = model.invoke(prompt)
if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write(result.content)


