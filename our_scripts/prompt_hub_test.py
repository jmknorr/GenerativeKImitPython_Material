# %% packages
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# %% Prompt Template
prompt_template = hub.pull("smithing-gold/assumption-checker")

prompt_template # print prompt template to see input variables

# %% model
MODEL_NAME = "openai/gpt-oss-120b"

model = ChatGroq(
    model=MODEL_NAME, 
    temperature=0, 
    api_key=os.getenv("GROQ_API_KEY")
    )

# %% chain
chain = prompt_template | model | StrOutputParser()

# %% invoke chain
user_query = "Es sollte ein Smartphone Verbot für Kinder unter 12 Jahren geben."

response = chain.invoke({"question": user_query})


# %%
print(response)

