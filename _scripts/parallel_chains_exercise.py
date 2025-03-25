#%%
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

#%% Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini")

# Create templates for different creative tasks
character_template = PromptTemplate.from_template(
    "Create a unique character name for a fantasy story set in {setting}"
)

backstory_template = PromptTemplate.from_template(
    "Write a compelling backstory for a character in a {genre} world."
)

quest_template = PromptTemplate.from_template(
    "Design an exciting quest for a hero in a {setting} world"
)

#%% Define parallel runnable to generate multiple story elements simultaneously
story_generator = RunnableParallel({
    "name": character_template | model | StrOutputParser(),
    "backstory": backstory_template | model | StrOutputParser(),
    "quest": quest_template | model | StrOutputParser()
})

# Function to run the parallel story generation
setting="floating sky islands"
genre="steampunk fantasy"
result = story_generator.invoke({
    "setting": setting,
    "genre": genre,
    "name": "John Doe"
})
    
    # Combine the generated elements
full_story = f"""
Story Elements Generator:

Character Name: {result['name']}

Backstory: {result['backstory']}

Epic Quest: {result['quest']}
"""


# %%
from rich.markdown import Markdown
from rich.console import Console

console = Console()

console.print(Markdown(full_story))

# %%
