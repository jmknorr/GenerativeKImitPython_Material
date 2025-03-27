#%% packages
import gensim

#%% download glove word embeddings for 6B
import gensim.downloader
glove_vectors = gensim.downloader.load("glove-wiki-gigaword-50")

#%% most similar words
glove_vectors.most_similar("king")

#%% get embedding for word
# calculate with vectors
#  Japan - Sushi + Bratwurst = ?
most_similar = glove_vectors["germany"] + glove_vectors["mussolini"] - glove_vectors["italy"]

# find the most similar word to a vector
glove_vectors.most_similar(most_similar)



