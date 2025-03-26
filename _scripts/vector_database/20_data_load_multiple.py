#%%
from langchain_community.document_loaders import  TextLoader
from langchain_community.document_loaders.markdown import  UnstructuredMarkdownLoader

from langchain_community.document_loaders.pdf import PyPDFLoader
# word loader
from langchain_community.document_loaders.word_document import Docx2txtLoader
import os
#%%
all_files = os.listdir("data/")
all_files

#%% iterate over all files and load them
docs = []
for file in all_files:
    file_path = os.path.join("data/", file)
    print(f"Loading {file_path}")
    if file.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        doc = loader.load()
    elif file.endswith(".md"):
        loader = UnstructuredMarkdownLoader(file_path)
        doc = loader.load()
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        doc = loader.load()
    elif file.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        doc = loader.load()
    else:
        print(f"Skipping {file} because it is not a supported file type")
    docs.extend(doc)
        


# %%
docs