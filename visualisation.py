#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, os
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#%%

# Read network edge list
G = nx.read_edgelist("info/kegg_human-edgelist/large/hsa00010.tsv")
nodes= G.nodes()

# Import gene expression data
expression_profile = pd.read_csv("data/raw/data_labelled.tsv", sep="\t")
# Convert row names from int to str
id = "1_1"
expression_profile.index = expression_profile.index.map(str)
features_list = [expression_profile.loc[gene, id] for gene in nodes]

plot = nx.draw(G,
                with_labels=True,
                node_size=1000,
                node_color = features_list,
                cmap="Reds")

plt.savefig("dump/hsa00010_1.png",
            dpi=150, bbox_inches='tight')
plt.show()

#features_list
#np.where(expression_profile["0_1"] == 3631.2)
#expression_profile.index[254]


#%%

# PCA
EMBEDDING_FILE = "data/embedding/gae/0_1-hsa00010.npy"
X = np.load(EMBEDDING_FILE)
X_reduced = PCA(n_components=3).fit_transform(X)

EMBEDDING_FILE = "data/embedding/gae/0_10-hsa00010.npy"
X1 = np.load(EMBEDDING_FILE)
X1_reduced = PCA(n_components=3).fit_transform(X1)

EMBEDDING_FILE = "data/embedding/gae/1_1-hsa00010.npy"
X2 = np.load(EMBEDDING_FILE)
X2_reduced = PCA(n_components=3).fit_transform(X2)

EMBEDDING_FILE = "data/embedding/gae/1_10-hsa00010.npy"
X3 = np.load(EMBEDDING_FILE)
X3_reduced = PCA(n_components=3).fit_transform(X3)

mpl.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=50, c="r")
ax.scatter(X1_reduced[:, 0], X1_reduced[:, 1], s=50, c="m")
ax.scatter(X2_reduced[:, 0], X2_reduced[:, 1], s=50, c="b")
ax.scatter(X3_reduced[:, 0], X3_reduced[:, 1], s=50, c="c")
#for x, y, label in zip(X_reduced[:, 0], X_reduced[:, 1], labels):
#    ax.text(x, y, label)
ax.set_ylim(-0.000020,0.000020)
ax.set_ylabel("PC2")
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlim(-0.6], 0.6)
ax.set_xlabel("PC1")
ax.set_xticks([])
ax.set_xticklabels([])
plt.tight_layout()
#plt.savefig("dump/hsa00010_4.png",
#            dpi=150, bbox_inches='tight')
plt.show()

# Stacked nodes
data_list = [X,X1,X2,X3]
X_stack = np.vstack(data_list)
X_stack_reduced = PCA(n_components=2).fit_transform(X_stack)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(X_stack_reduced[0:15, 0], X_stack_reduced[0:15, 1], s=50, c="r")
ax.scatter(X_stack_reduced[15:30, 0], X_stack_reduced[15:30, 1], s=50, c="m")
ax.scatter(X_stack_reduced[30:45, 0], X_stack_reduced[30:45, 1], s=50, c="b")
ax.scatter(X_stack_reduced[45:60, 0], X_stack_reduced[45:60, 1], s=50, c="c")
plt.show()

# 3D PCA Plot
#fig, ax = plt.subplots(1, 1, figsize=(5, 3))
#ax = Axes3D(fig, elev=-150, azim=110)
#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2])
#
#for x, y, z, label in zip(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], labels):
#    ax.text(x, y, z, label)
#
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])
#plt.show()

#%%

# TSNE visualisation
X_tsne = TSNE(n_components=2, perplexity=20.0).fit_transform(X_stack)
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X_tsne[:,0], X_tsne[:,1])

#for x, y, label in zip(X_embedded[:,0], X_embedded[:,1], labels):
#    ax.text(x, y, label)

#%%

FILE_DIR = "data/embedding/gae/"
files = os.listdir(FILE_DIR)
data_list = [np.load(os.path.join(FILE_DIR, file)) for file in files]
X_stack = np.vstack(data_list)

#%%

X_stack_reduced = PCA(n_components=3).fit_transform(X_stack)
X_stack_reduced.shape

#ax.scatter(X_stack_reduced[0:15, 0], X_stack_reduced[0:15, 1], s=50, c="r")
#ax.scatter(X_stack_reduced[15:30, 0], X_stack_reduced[15:30, 1], s=50, c="m")
#ax.scatter(X_stack_reduced[30:45, 0], X_stack_reduced[30:45, 1], s=50, c="m")
#
#ax.scatter(X_stack_reduced[2700:2715, 0], X_stack_reduced[2700:2715, 1], s=50, c="b")
#ax.scatter(X_stack_reduced[2715:2730, 0], X_stack_reduced[2715:2730, 1], s=50, c="c")
#ax.scatter(X_stack_reduced[2730:2735, 0], X_stack_reduced[2730:2735, 1], s=50, c="c")
#plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
PATHWAY_SIZE=15
ax.scatter(X_stack_reduced[0:179*PATHWAY_SIZE, 0],
           X_stack_reduced[0:179*PATHWAY_SIZE, 1],
           cmap="Reds",
           c=np.arange(179*PATHWAY_SIZE),
           s=50)
ax.scatter(X_stack_reduced[179*PATHWAY_SIZE:286*PATHWAY_SIZE, 0],
           X_stack_reduced[179*PATHWAY_SIZE:286*PATHWAY_SIZE, 1],
           cmap="Blues",
           c=np.arange(107*PATHWAY_SIZE),
           s=50)

ax.set_ylabel("PC2")
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_xlabel("PC1")
ax.set_xticks([])
ax.set_xticklabels([])
plt.tight_layout()
plt.show()

#
#plt.savefig("dump/hsa00010_all.png",
#            dpi=150, bbox_inches='tight')

#for i in range(len(data_list)):
#    start_index = i * 15
#    end_index = (i+1) * 15
#    if i < 179 :
#        ax.scatter(X_stack_reduced[start_index:end_index, 0],
#                   X_stack_reduced[start_index:end_index, 1],
#                   cmap="Reds",
#                   c=np.arange(15))
#    else:
#        ax.scatter(X_stack_reduced[start_index:end_index, 0],
#                   X_stack_reduced[start_index:end_index, 1],
#                   cmap="Blues",
#                   c=np.arange(107))



#%%
# Sum nodes into graphs
graph_list = []

for i in range(len(data_list)):
    start_index = i * 15
    end_index = (i+1) * 15
    graph_dim = np.mean(X_stack_reduced[start_index:end_index, :], axis=0)[np.newaxis,:]
    graph_list.append(graph_dim)

X_graph_stack = np.vstack(graph_list)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.scatter(X_graph_stack[0:179, 0],
           X_graph_stack[0:179, 1], c="r")

ax.scatter(X_graph_stack[179:286, 0],
           X_graph_stack[179:286, 1], c="b")

plt.show()
#%%

# 3D PCA Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = Axes3D(fig, elev=-150, azim=110)
for i in range(len(data_list)):
    start_index = i * 15
    end_index = (i+1) * 15
    if i < 179 :
        ax.scatter(X_stack_reduced[start_index:end_index, 0],
                   X_stack_reduced[start_index:end_index, 1],
                   X_stack_reduced[start_index:end_index, 2],
                   c="r")
    else:
        ax.scatter(X_stack_reduced[start_index:end_index, 0],
                   X_stack_reduced[start_index:end_index, 1],
                   X_stack_reduced[start_index:end_index, 2],
                   c="b")


#%%

# TSNE visualisation
#X_tsne = TSNE(n_components=2, perplexity=20.0).fit_transform(X_stack)

fig, ax = plt.subplots(figsize=(6,4))
for i in range(len(data_list)):
    start_index = i * 15
    end_index = (i+1) * 15
    if i < 179 :
        ax.scatter(X_tsne[start_index:end_index, 0],
                   X_tsne[start_index:end_index, 1], c="r")
    else:
        ax.scatter(X_tsne[start_index:end_index, 0],
                   X_tsne[start_index:end_index, 1], c="b")
plt.show()

#%%

with open('dump/cnn_history-node.pkl', 'rb') as file:
    cnn_node = pickle.load(file)

cnn_hist["acc"]
cnn_node["acc"]
