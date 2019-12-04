#%%
import pickle, os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

os.chdir("/home/dblux/projects/phd/cs6216")

#%%
# Load genes data
GENE_FILE = "data/gene/genes_20.tsv"
#GENE_FILE = "data/gene/genes_100.tsv"
#GENE_FILE = "data/gene/genes_1983.tsv"
#GENE_FILE = "data/gene/genes_all.tsv"
#PATHWAY_FILE = "data/pathway/X_pathway.tsv"

X = np.loadtxt(GENE_FILE, delimiter='\t', dtype=np.float32, ndmin=2)
y = np.concatenate((np.repeat(0, 179), np.repeat(1, 107))).astype("int16")

# Random stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=1,
                                                    test_size=0.25)

#names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Multi-layer perceptron"]
print("Class proportion (train):", sum(y_train == 0)/len(y_train))
print("Class proportion (test):", sum(y_test == 0)/len(y_test))

#%%

print(X_train.shape)

#%%

classifiers = [
    LogisticRegression(C=1, solver="liblinear"),
    SVC(kernel="linear",
        gamma="scale",
        C=1,
        class_weight={0: 0.4, 1: 0.6},
        probability=True),
    SVC(kernel="poly",
        gamma="scale",
        C=1,
        class_weight={0: 0.4, 1: 0.6},
        probability=True),
    RandomForestClassifier(n_estimators = 10),
    MLPClassifier(hidden_layer_sizes=(10,10), alpha=0.1)]

acc, y_prob, y_predicted = [], [], []
 # iterate over classifiers
for clf in classifiers:
    # Instantiate classifier
    classifier = clf
    # Train the classifier object
    print("Training model...")
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

#%%
# Dump performance metrics
with open('dump/100_genes.pkl', 'wb') as file:
    pickle.dump(performance, file)

#%%
# Load dumped data    
#with open('dump/100_genes.pkl', 'rb') as file:
#    all_genes = pickle.load(file)

acc, y_prob, y_predicted, y_test = performance
print(acc)

for y_i in y_predicted:
    print(classification_report(y_test, y_i))
    print(confusion_matrix(y_test, y_i))

#%%
# Optimise hyperparameters
parameters = {'kernel':('linear', 'rbf', 'poly'),
              'C':(0.01, 1, 10),
              'class_weight':({0:0.5, 1: 0.5},
                              {0:0.4, 1: 0.6})
              }
svc = SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)
SVC_results = clf.cv_results_
#hyperp_results.keys()

print("SVM:")
for i in zip(SVC_results["params"], SVC_results["mean_test_score"]):
    print(i)

# Optimise hyperparameters
parameters = {'C':[0.01, 1, 10]}
log_reg = LogisticRegression(solver="liblinear")
clf1 = GridSearchCV(log_reg, parameters, cv=5)
clf1.fit(X_train, y_train)
logreg_results = clf1.cv_results_
#hyperp_results.keys()
print("Logistic regression:")
for i in zip(logreg_results["params"], logreg_results["mean_test_score"]):
    print(i)

#%%

parameters = {'n_estimators': (300, 600, 800)}
rf = RandomForestClassifier()
clf2 = GridSearchCV(rf, parameters, cv=5)
clf2.fit(X_train, y_train)
rf_results = clf2.cv_results_

print("RF:")
for i in zip(rf_results["params"], rf_results["mean_test_score"]):
    print(i)

parameters = {'hidden_layer_sizes': ((500,200,200,100,100),
                                     (500,200,100))}
mlp = MLPClassifier(alpha=0.1)
clf3 = GridSearchCV(mlp, parameters, cv=5)
clf3.fit(X_train, y_train)
mlp_results = clf3.cv_results_

print("MLP:")
for i in zip(mlp_results["params"], mlp_results["mean_test_score"]):
    print(i)

#%%
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#%%
# Load data set
#digits = datasets.load_digits()
#X = digits.data
#y = digits.target
#class_names = list(digits.target_names.astype("str"))

#%% VISUALISATION
# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions

#X_reduced = PCA(n_components=20).fit_transform(X)

type(X_reduced)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2])
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1, projection='3d')

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

#%%
