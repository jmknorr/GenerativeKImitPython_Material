from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import pprint

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

messages = [
    ("system", """You are a travel planner. Given a destination, you return a 
    JSON object strictly following this structure: 
    { 
        "destination details" : "some key infos about the destination", 
        "cultural insights" : "list of some insights about the destination", 
        "budget" : "what would be the average daily budget in € for the trip", 
        "transportation" : "what is the most appropriate transportation mode for the trip"
    }"""),
    ("user", "Please plan my trip to {destination}"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

MODEL_NAME = "llama3-8b-8192"
model = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY)
parser = SimpleJsonOutputParser()
chain = prompt_template | model | parser

user_input = {"destination": "Leipzig"}
res = chain.invoke(user_input)

# check if res is a dict
if isinstance(res, dict):
    for key, value in res.items():
        print(f"{key.upper()}: {value}\n")
else:
    print("Error: the result is not a dictionary.")
    print(res.content)