# %% packages
# %% packages
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# %%
url = "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=OJ:L_202401689"
loader = WebBaseLoader(url)

# wiki_loader = WikipediaLoader(query="EU AI Act", lang="de", load_max_docs=3)


# %%
data = loader.load()
print(data)

# wiki_data = wiki_loader.load()
# wiki_data

# %%
print(data[0].page_content[200:500])

# wiki_data[0].metadata

# %% data chunking (starres splitting basierend auf chracter)
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n",
)

data_splitted = splitter.split_documents(data)
print(len(data_splitted))

chunk_sizes = [len(d.page_content) for d in data_splitted]
print(chunk_sizes)

# %% rekursives chunking (weicheres splitting basierend auf mehreren separatoren)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

data_splitted = splitter.split_documents(data)

chunk_sizes = [len(d.page_content) for d in data_splitted]
print(chunk_sizes)
# %% visualize chunk sizes
# import seaborn as sns

# sns.histplot(chunk_sizes)

# %% embedding
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)

from langchain_openai.embeddings import OpenAIEmbeddings

sentences = ["This is an example sentence", "Each sentence is converted"]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

print(embeddings_model)

# %% database
from langchain_chroma import Chroma

db = Chroma(embedding_function=embeddings_model)

db.add_documents(data_splitted)