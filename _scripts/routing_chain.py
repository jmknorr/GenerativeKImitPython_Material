#%% packages    
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain_openai import ChatOpenAI


#%% load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from pydantic import BaseModel, Field
# %% implement patientUrgency with predefined levels "low", "medium", "high"
class PatientUrgency(BaseModel):
    """Output for patient urgency."""
    urgency: str = Field(..., description="Urgency of the patient's condition", pattern="^(low|medium|high)$")
    
    

# %% implement chain for urgency classification
model = ChatOpenAI(model="gpt-4o-mini")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that classifies the urgency of a patient's condition."),
        ("user", "Here is the patient description: {patient_description}. Please return a JSON object with the key 'urgency' of the patient's condition in JSON format. Level can be 'low', 'medium', 'high'.")
    ]
)
chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=PatientUrgency)
    

# %%
chain.invoke({"patient_description": "I am bleeding a lot."})

# %%
