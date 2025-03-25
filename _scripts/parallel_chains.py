#%% 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
# %% prompt template
messages = [
    ("system", "You are a mathematical genius and solve riddles."),
    ("user", "Please solve the riddle: {riddle}")
]
prompt_template = ChatPromptTemplate(messages=messages)

#%% models
model_a = ChatOpenAI(model= "gpt-4o-mini")
model_b = ChatGroq(model="llama-3.1-8b-instant")
model_c = ChatGroq(model="llama-3.3-70b-versatile")

#%% define parallel chain
mapped_chain = RunnableParallel(
    model_a = prompt_template | model_a | StrOutputParser(),
    model_b = prompt_template | model_b | StrOutputParser(),
    model_c = prompt_template | model_c | StrOutputParser()
)



#%% inference
user_prompt = """
Find a solution to the game of 24 with the values 4, 7, 8, 8.
"""
res = mapped_chain.invoke({"riddle": user_prompt})

#%% check the result 
from pprint import pprint
pprint(res, width=50)
# %%
