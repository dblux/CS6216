#!/usr/bin/env python
# coding: utf-8

import pickle, os
import numpy as np
from sklearn.decomposition import PCA

EMBEDDING_FILE = "data/embedding/X_embedding.npy"
X = np.load(EMBEDDING_FILE)
print(X.shape)
