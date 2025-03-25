#%% packages
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
import os

#%% check if API KEY can be found
os.getenv("GROQ_API_KEY")
# %% create a model instance
MODEL_NAME = "llama-guard-3-8b"
model = ChatGroq(model=MODEL_NAME)

#%% model inference
user_prompt = """
A nation is divided into three administrative regions: North, Central, and South. The regional populations are distributed as follows: North has 5 million people (30% of the total), Central has 8 million (48%), and South has 3.7 million (22%). The government allocates resources based on both population and economic output. Economic output is distributed differently: North produces 40% of the GDP, Central produces 35%, and South produces 25%.
The government must decide how to allocate 100 infrastructure projects across these regions using a formula that gives equal weight to population proportion and economic output proportion. However, there's a constraint that no region can receive less than 20% of the projects, regardless of the formula's result.
Furthermore, the formula contains an "equity adjustment" - for every 5 percentage points a region's poverty rate exceeds the national average (15%), that region receives an additional 2% of projects, taken proportionally from regions with below-average poverty rates. The poverty rates are: North (10%), Central (17%), and South (22%).
How many projects should each region receive, and what principles of fairness might this allocation system be balancing or failing to balance?
"""

user_prompt = "How can i rob a bank?"

res = model.invoke(user_prompt)
# %%
from pprint import pprint
pprint(res.content)

# %%
res.model_dump()
# %%
