import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.fast_gnp_random_graph(n=100, p=0.5)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# Look for most similar nodes
model.wv.most_similar('2')  # Output node names are always strings

# Save embeddings for later use
EMBEDDING_FILENAME = "embeddings.txt"
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use

model.save(EMBEDDING_MODEL_FILENAME)

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder

edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Look for embeddings on the fly - here we pass normal tuples
edges_embs[('1', '2')]
''' OUTPUT
array([ 5.75068220e-03, -1.10937878e-02,  3.76693785e-01,  2.69105062e-02,
       ... ... ....
       ..................................................................],
      dtype=float32)
'''

# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()

# Look for most similar edges - this time tuples must be sorted and as str
edges_kv.most_similar(str(('1', '2')))

# Save embeddings for later use
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)

#%%

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn import datasets

# Remove column header from edge list! 
hsa05332 = nx.read_edgelist("../diff_expr/data/kegg_human-edgelist/hsa05332.tsv")
hsa00010 = nx.read_edgelist("../diff_expr/data/kegg_human-edgelist/hsa00010.tsv")

# Plot graph
nx.draw(hsa05332, with_labels=True)
nx.draw(hsa00010, with_labels=True)
plt.show()

# List of edges 
hsa05332.edges()
# Number of edges
hsa05332.size()
# Number of nodes
hsa00010.__len__()

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(hsa00010, dimensions=64, walk_length=80,
                    num_walks=200, workers=4, p=1, q=1)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

data = model.wv.__dict__
labels = data["index2word"]
data_arr = data["vectors"]
data_arr.shape

# Node2Vec
# How to change p and q parameters???
# p = 1 and q = 1 is equal to DeepWalk
# Better to perform more BFS

#%% VISUALISATION
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions

X_reduced = PCA(n_components=3).fit_transform(data_arr)
X_reduced1 = PCA(n_components=3).fit_transform(data_arr1)

fig = plt.figure(1, figsize=(15, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2])

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter(X_reduced1[:, 0], X_reduced1[:, 1], X_reduced1[:, 2])
plt.show()

#ax1 = Axes3D(fig, elev=-150, azim=110)
#ax2 = Axes3D(fig, elev=-150, azim=110)


for x, y, z, label in zip(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], labels):
    ax1.text(x, y, z, label)

ax1.set_xlabel("1st eigenvector")
ax1.w_xaxis.set_ticklabels([])
ax1.set_ylabel("2nd eigenvector")
ax1.w_yaxis.set_ticklabels([])
ax1.set_zlabel("3rd eigenvector")
ax1.w_zaxis.set_ticklabels([])
plt.show()


plt.clf()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
ax1.scatter(X_reduced[:, 0], X_reduced[:, 1])
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1])

# TSNE visualisation
X_embedded = TSNE(n_components=2, perplexity=20.0).fit_transform(data_arr)
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(X_embedded[:,0], X_embedded[:,1])

for x, y, label in zip(X_embedded[:,0], X_embedded[:,1], labels):
    ax.text(x, y, label)

# Save embeddings for later use
EMBEDDING_FILENAME = "hsa05532_n2v.txt"
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
model.save(EMBEDDING_MODEL_FILENAME)

