#%% pakete
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
# import seaborn as sns
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma
load_dotenv(find_dotenv())
 
# %% data loading
eu_ai_act_url = "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=OJ:L_202401689"
loader = WebBaseLoader(eu_ai_act_url)
docs = loader.load()
docs[0].metadata
  
# %% data chunking (Struktur-basierter Ansatz)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=["KAPITEL ", "Artikel ", "\n\n", "\n", " ", ""])
docs_splitted = splitter.split_documents(docs)
chunk_sizes = [len(doc.page_content) for doc in docs_splitted]
chunk_sizes

#%% embedding
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# %% database
db = Chroma(embedding_function=embeddings_model, persist_directory="ai_act")
db.add_documents(documents=docs_splitted)

# %%