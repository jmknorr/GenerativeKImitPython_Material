#%% packages
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
import numpy as np
import seaborn as sns
#%% NATIVE MODEL
#%% model instance
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

#%%
sentences = ["This is an example sentence", "Each sentence is converted"]
embeddings = model.encode(sentences)
#%%
print(embeddings.shape)

#%% LANGCHAIN PORT
embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
embeddings_model.embed_query(sentences[0])

#%% sentence similarity
sentences = [
    'The cat lounged lazily on the warm windowsill.',
    'A feline relaxed comfortably on the sun-soaked ledge.',
    'The kitty reclined peacefully on the heated window perch.',
    'Quantum mechanics challenges our understanding of reality.',
    'The chef expertly julienned the carrots for the salad.',
    'The vibrant flowers bloomed in the garden.',
    'Las flores vibrantes florecieron en el jardín. ',
    'Die lebhaften Blumen blühten im Garten.'
]

sentence_embeddings = embeddings_model.embed_documents(sentences)

#%% 
corr = np.corrcoef(sentence_embeddings)
sns.heatmap(corr, annot=True, fmt=".1f", xticklabels=sentences, yticklabels=sentences)
# %%
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
#%%
sentence_embeddings = model.encode(sentences=sentences)
#%%
corr = np.corrcoef(sentence_embeddings)
sns.heatmap(corr, annot=True, fmt=".1f", xticklabels=sentences, yticklabels=sentences)
# %%
