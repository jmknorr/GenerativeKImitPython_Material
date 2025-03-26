#%% packages
from langchain.document_loaders import TextLoader
import os
import sys
__file__ = os.path.abspath(sys.argv[0])

#%%
file_path_script = os.path.abspath(os.__file__)
current_dir = os.path.dirname(file_path_script)
current_dir
# %% instance of Document loader
file_path_rel = "data/Romeo_and_Julia.txt"
file_path_abs = os.path.join(current_dir, file_path_rel)
loader = TextLoader(file_path=file_path_abs, encoding="utf-8")


#%% extract content from book
docs = loader.load()

#%%
docs[0].page_content
# %%