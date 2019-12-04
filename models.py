#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle, os
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report


# In[56]:


# Pre-processing embeddings
EMBEDDING_DIR = "data/embedding/gae/"
embedding_files = os.listdir(EMBEDDING_DIR)
embedding_files.sort()


# In[107]:


num_files = len(embedding_files)
print(num_files)
for i in range(286):
    print(embedding_files[i*158])


# In[65]:


flattened_data = [np.load(os.path.join(EMBEDDING_DIR, file)).reshape(1,-1) for file in embedding_files]


# In[92]:


data_list = []
for indv in range(286):
    start_index = indv * 158
    end_index = (indv + 1) * 158
    print(start_index, end_index)
    data_list.append(np.hstack(flattened_data[start_index:end_index]))


# In[104]:


X = np.vstack(data_list)
print(X.shape)
# Save processed embedding data (286x104800)
output_fpath = "data/embedding/X_embedding.npy"
# np.save(output_fpath, X)


# In[117]:


# Embeddings
X = np.load(output_fpath)
print(X.shape)
# Labels y
y = np.concatenate((np.repeat(0, 179), np.repeat(1, 107))).astype("int16")

# Random stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1,
                                                    test_size=0.25)


# In[130]:


print("Class proportion (train):", sum(y_train == 0)/len(y_train))
print("Class proportion (test):", sum(y_test == 0)/len(y_test))

classifiers = [
#     LogisticRegression(C=1, solver="liblinear"),
#     SVC(kernel="linear", C=1, probability=True),
    SVC(kernel="poly", gamma="scale", C=1, probability=True),
    RandomForest(),
    MLPClassifier(hidden_layer_sizes=(10000,1000,100), alpha=0.1)]

acc, y_prob, y_predicted = [], [], []
 # iterate over classifiers
for clf in classifiers:
    # Instantiate classifier
    classifier = clf
    # Train the classifier object
    print("Training model:", classifier.__class__)
    classifier.fit(X_train, y_train)
    # Evaluate classifier
    acc.append(classifier.score(X_test, y_test))
    if hasattr(classifier, "decision_function"):
        print("Has decision function...")
        y_prob.append(classifier.decision_function(X_test))
    else:
        y_prob.append(classifier.predict_proba(X_test))
    y_predicted.append(classifier.predict(X_test))

performance = [acc, y_prob, y_predicted, y_test]
print("Done!")


# In[126]:


# Dump performance metrics
with open('dump/logreg_linear_smallmlp.pkl', 'wb') as file:
    pickle.dump(performance, file)


# In[124]:


acc, y_prob, y_predicted, y_test = performance
print(acc)

for y_i in y_predicted:
    print(classification_report(y_test, y_i))
    print(confusion_matrix(y_test, y_i))


# In[131]:





# In[ ]:




