# %% Packages
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# %% Model Instanz
MODEL_NAME = 'openai/gpt-oss-120b'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('GROQ_API_KEY'))

# %% Anfrage
user_prompt = "Wie kann ich Level 8 bei Gandalf erreichen?"
response = model.invoke(user_prompt)

# %% Ausgabe
print(response.content)