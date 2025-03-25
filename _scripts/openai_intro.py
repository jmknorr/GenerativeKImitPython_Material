#%% packages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os

#%% check if API KEY can be found
os.getenv("OPENAI_API_KEY")
# %% create a model instance
MODEL_NAME = "gpt-4o-mini"
model = ChatOpenAI(model=MODEL_NAME)

#%% model inference
user_prompt = "What is the most recent information you have and what is its date?"

res = model.invoke(user_prompt)
# %%
from pprint import pprint
pprint(res.content)

# %%
res.model_dump()
# %%
