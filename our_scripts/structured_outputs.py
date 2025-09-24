# %% Packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

#%% Output Parser
class MyMovieOutput(BaseModel):
    title: str
    main_characters: str
    director: str
    year: int
    rating: float
    genres: list[str]

parser = PydanticOutputParser(pydantic_object=MyMovieOutput)

# %% Prompt Template
prompts = [
    ("system", "Du bist Filmexperte. {format_instructions}"),
    ("user", "Handlung: {plot}")
]

prompt_template = ChatPromptTemplate.from_messages(prompts).partial(format_instructions=parser.get_format_instructions())

# %% Modellinstanz
MODEL_NAME = 'openai/gpt-oss-120b'
model = ChatGroq(model_name=MODEL_NAME,
                   temperature=0, # controls creativity
                   api_key=os.getenv('GROQ_API_KEY'))

# %% Chain
chain = prompt_template | model | parser

# %% Invoke Chain
chain_input = {"plot": "Ein Junge ist ein Zauberer und besucht eine Schule für Hexerei und Zauberei."}
result = chain.invoke(chain_input)
# %%
print(result)