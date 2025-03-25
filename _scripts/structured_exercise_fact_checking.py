#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import SimpleJsonOutputParser, JsonOutputParser
from pydantic import BaseModel
#%% target output
# { "title": "", "author": "...", "summary":}
class PlaceOutput(BaseModel):
    places: str
    
class ScoreOutput(BaseModel):
    score: int
    reason: str

# %%
messages = [
    ("system", "You are a city expert and deliver key places in a city."),
    ("user", """Please provide key information on a city: <{city}>. Return the result as JSON following this template: 
    {{
        "places": "most relevant places in the city formatted as a string, not a list", 
    }}""")
]
prompt_template_places = ChatPromptTemplate.from_messages(messages)
prompt_template_places

#%%
messages = [
    ("system", "You are a city expert and fact check on the places in a city. provide a score between 0 and 100."),
    ("user", """Please provide key information on places in a city: <{places}>. Return the result as JSON following this template: 
    {{
        "score": "score between 0 and 100", 
        "reason": "reason for the score"
    }}""")
]
prompt_template_score = ChatPromptTemplate.from_messages(messages)

# %%
# prompt_template.invoke({"topic": "2025"})
# %% model
MODEL_NAME = "gemma2-9b-it"
model = ChatGroq(model=MODEL_NAME)


MODEL_NAME_FACT_CHECK = "gpt-4o-mini"
model_fact_check = ChatOpenAI(model=MODEL_NAME_FACT_CHECK)

#%% chain
# chain = prompt_template | model | JsonOutputParser(pydantic_object=BookOutput)
chain = prompt_template_places | model | SimpleJsonOutputParser(pydantic_object=PlaceOutput) | prompt_template_score | model_fact_check | SimpleJsonOutputParser(pydantic_object=ScoreOutput)

# %% chain inference
user_input = {"city": "Fürstenwalde/Spree"}
res = chain.invoke(user_input)
res
#%% show result in console via rich, show only 50 characters per line
res
# %%
