#%% packages
from langchain_community.document_loaders import WikipediaLoader
import seaborn as sns
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings

#%% load wikipedia
query = "Large Language Model"
loader = WikipediaLoader(query=query, load_max_docs=10)
docs = loader.load()

# %%
docs


# %% Fixed-size
CHUNK_SIZE = 5 * 200 # 5 chars/word, 100 words
splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap = CHUNK_SIZE / 4, separator="")
docs_splitter_fixed = splitter.split_documents(docs)
len(docs_splitter_fixed)
# %%
from pprint import pprint
pprint(docs_splitter_fixed[1].page_content)
# %% check the sizes of the chunks
docs_length = [len(doc.page_content) for doc in docs_splitter_fixed]
sns.histplot(docs_length, bins=20)

#%% Recursive Character Text Splitter
splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_SIZE / 8)
docs_splitter_recursive = splitter.split_documents(docs)
len(docs_splitter_recursive)


# %% distribution of chunk sizes
docs_length = [len(doc.page_content) for doc in docs_splitter_recursive]
sns.histplot(docs_length, bins=20)

#%% study very small chunks
# get index of chunks with less than 100 chars
small_chunks = [i for i, length in enumerate(docs_length) if length < 100]
small_chunks

#%% print the first 10 small chunks
for i in range(3):
    print(docs_splitter_recursive[small_chunks[i]].page_content)
    print("-"*30)


#%% Semantic Chunking
splitter = SemanticChunker(embeddings=OpenAIEmbeddings()
                           )
docs_splitter_semantic = splitter.split_documents(docs[:1])
# %%
len(docs_splitter_semantic)
# %%
pprint(docs_splitter_semantic[2].page_content)



# %% embed the documents
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

#%% 
docs_embeddings = model.embed_documents([doc.page_content for doc in docs_splitter_recursive])

#%%
len(docs_embeddings)

#%%
len(docs_embeddings[0])

#%% use FAISS
import numpy as np
embeddings_model = OpenAIEmbeddings()
docs_texts = [doc.page_content for doc in docs_splitter_recursive]
docs_embeddings_openai = embeddings_model.embed_documents(docs_texts)

docs_embeddings_np = np.array(docs_embeddings_openai)
docs_embeddings_np.shape

#%% create FAISS
text_embedding_pairs = zip(docs_texts, docs_embeddings_np)

index_openai_emb = FAISS.from_embeddings(text_embedding_pairs, embeddings_model)

#%%
user_query = "What is DeepSeek?"
index_openai_emb.similarity_search(user_query)

#%%
retriever_faiss = index_openai_emb.as_retriever()
retriever_faiss.invoke(user_query)

#%% store info in vector DB (Chroma)

db = Chroma(persist_directory="db_wikipedia_openai", collection_name="llm", embedding_function=embeddings_model)

#%%
db.add_documents(docs_splitter_recursive)

#%%
db.get().keys()

#%%
len(docs_splitter_recursive)

#%% Retriever
retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.3},
        include_metadata=True,
        include_distances=True
    )

#%% 
query = "Which LLM was developed by Meta?"
res = retriever.invoke(query)
res
# Calculate the cosine similarity score
# res[0].metadata.get("distance")


#%%
res[0].keys()

#%%
def rag(user_query: str, model_name="llama-3.3-70b-versatile") -> str:
    """implement a simple RAG system

    Args:
        user_query (str): The user's question

    Returns:
        str: The RAG answer
    """
    # 1. Retrieval
    res = retriever.invoke(user_query)
    context = "; ".join([doc.page_content for doc in res])
    print(context)
    
    # 2. Augmentation
    messages = [
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
        ("user", "Question: {question}, Retrieved Context: {retrieved_context}")
    ]
    prompt_template = ChatPromptTemplate(messages=messages)
    llm = ChatGroq(model=model_name)
    chain = prompt_template | llm | StrOutputParser()
    llm_response = chain.invoke({'question': user_query, 'retrieved_context': context})
    return llm_response
    
    
# %%
user_query = "What is DeepSeek?"
rag(user_query=user_query)
# %%
