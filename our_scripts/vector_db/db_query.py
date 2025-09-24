#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

#%% embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(persist_directory="ai_act", embedding_function=embeddings_model)


# %% query database
retriever = db.as_retriever( 
    search_kwargs={
        "k": 4,
        "score_threshold": 0.0,     # je höher desto ähnlicher
    },
    search_type="similarity_score_threshold"
)

user_query = "Welche Regelungen betreffen Hochrisiko-Systeme?"

# results = retriever.invoke(user_query)

# %% show results
# from pprint import pprint

# for r in results:
#     pprint(r.page_content, width=50)
#     print("-----")

# %% RAG
from pprint import pprint

def rag(user_query: str) -> str:
    """" erstellt die RAG-Antwort auf Basis des Kontextes von der Vektor-DB
    user_query = Frage des Users 
    Return: Antwort des Modells
    """
    # 1. Retrieval
    context = retriever.invoke(user_query)
    context_string = "; ".join([doc.page_content for doc in context]) 
    
    # 2. Augmentierung
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein Rechtsexperte, der Fragen beantwortet. Beantworte ausschließlich auf Basis des bereitsgestellten Kontextes {context}. Wenn der Kontext die Beantwortung der Frage nicht zulässt, antworte mit 'Das weiß ich nicht'."),
        ("user", "Frage des Nutzers: {user_query}; Kontext: {context}"),
    ])

    # 3. Generierung
    MODEL_NAME = "openai/gpt-oss-120b"
    model = ChatGroq(
        model=MODEL_NAME, 
        temperature=0
    )
    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({
        "user_query": user_query,
        "context": context_string
    })
    return response

pprint(rag(user_query=user_query), width=50)

# %%
