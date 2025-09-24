# %% Pakete
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# %% Prompt Template 
system_prompt = "You are an AI assistant that translates English to another language."
user_prompt = "Translate the following English text to {target_language}: {text}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", user_prompt),
])

# %% Modell Instanz
MODEL_NAME = 'openai/gpt-oss-120b'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=0.5, # controls creativity
                   api_key=os.getenv('GROQ_API_KEY'))

# %% Chain
chain = prompt_template | model | StrOutputParser()

# %% Chain ausführen
chain.invoke(input={"target_language": "German", "text": "Hello, how are you?"})