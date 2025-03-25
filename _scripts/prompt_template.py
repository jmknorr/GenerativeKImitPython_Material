#%%
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from rich.markdown import Markdown
from rich.console import Console
from langchain_core.output_parsers import StrOutputParser
# %%
messages = [
    ("system", "Du bist ein sarkastischer Wahrsager. Bitte erstelle eine Wahrsagung mit viel schwarzem Humor zu dem vorgegebenen Thema."),
    ("user", "Sag mir die Zukunft zum Thema: {topic}")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt_template
# %%
# prompt_template.invoke({"topic": "2025"})
# %% model
MODEL_NAME = "llama-3.3-70b-versatile"
model = ChatGroq(model=MODEL_NAME)

#%% chain
chain = prompt_template | model | StrOutputParser()

# %% chain inference
user_input = {"topic": "2026"}
res = chain.invoke(user_input)

#%% show result in console via rich, show only 50 characters per line
console = Console()
console.print(Markdown(res), end="\n\n")
# %%
