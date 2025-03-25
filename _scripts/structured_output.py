#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import SimpleJsonOutputParser, JsonOutputParser
from pydantic import BaseModel
#%% target output
# { "title": "", "author": "...", "summary":}
class BookOutput(BaseModel):
    title: str
    author: str
    year_published: int
    summary: str


# %%
messages = [
    ("system", "You are a book expert and deliver key information on specific books."),
    ("user", """Please provide key information on the book: <{book}>. Return the result as JSON following this template: 
    {{
        "title": str, 
        "author": str, 
        "year_published": int, 
        "summary": str
    }}""")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt_template
# %%
# prompt_template.invoke({"topic": "2025"})
# %% model
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)

#%% chain
# chain = prompt_template | model | JsonOutputParser(pydantic_object=BookOutput)
chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=BookOutput)

# %% chain inference
user_input = {"book": "1984"}
res = chain.invoke(user_input)

#%% show result in console via rich, show only 50 characters per line
res
# %%
parser = JsonOutputParser(pydantic_object=BookOutput)
prompt = f"""
Generate the output in the following JSON format:
{parser.get_format_instructions()}
"""
prompt
# %%
