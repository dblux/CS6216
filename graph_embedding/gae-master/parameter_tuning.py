import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Set working directory
os.chdir("/home/dblux/projects/phd/cs6216/graph_embedding/gae-master")
# Import gene expression data
expression_profile = pd.read_table("../../data/raw/data_labelled.tsv")
# Convert row names from int to str
expression_profile.index = expression_profile.index.map(str)
id_list = list(expression_profile)
# List large pathway files
KEGG_DIR = "../../info/kegg_human-edgelist/large/"
fname = os.listdir(KEGG_DIR)

# Initialise lists
file_list, num_edges_list, num_nodes_list, auc_list, ap_list = [], [], [], [], []
# Read graph as an undirected simple graph!
# Read as OrderedGraph if using python < 3.6
for file in fname[1:50]:
    fpath = os.path.join(KEGG_DIR, file)
    G = nx.read_edgelist(fpath)
    edges_num = len(G.edges())
    # Creates ordered list of nodes
    nodes = G.nodes()
    nodes_num = len(nodes)
    #nx.draw(G)
    #plt.show()
    
    # Create adj matrix from graph
    # adj is adjacency matrix: scipy sparse csr matrix
    adj = nx.adjacency_matrix(G)

# Loop through all sample IDs for each pathway
#    for id in id_list:
#        print(id)
    id = "0_1"
    # Fetch features using ordered nodes
    features_list = [expression_profile.loc[gene, id] for gene in nodes]
    # features is a scipy sparse lil matrix
    features = sp.lil_matrix(features_list)
    features = features.reshape(len(features_list),1)
  
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
    try:
        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    except:
        print("Error:", file)
        continue
    
    adj = adj_train
    
    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless
    
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }
    
    num_nodes = adj.shape[0]
    
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
    
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)
    
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    acc_val = []
    
    def get_roc_score(edges_pos, edges_neg, emb=None):
        if emb is None:
            feed_dict.update({placeholders['dropout']: 0})
            emb = sess.run(model.z_mean, feed_dict=feed_dict)
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
    
        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(adj_orig[e[0], e[1]])
    
        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(adj_orig[e[0], e[1]])
    
        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
    
        return roc_score, ap_score
           
    cost_val = []
    acc_val = []
    val_roc_score = []
    
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    # Train model
    for epoch in range(2000):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]
    
#        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
#        val_roc_score.append(roc_curr)
    
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy),
              "time=", "{:.5f}".format(time.time() - t))
    
    print("Optimization Finished!")
    
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
    # Append to lists
    file_list.append(file)
    num_nodes_list.append(nodes_num)
    num_edges_list.append(edges_num)
    auc_list.append(roc_score)
    ap_list.append(ap_score)
    print(file)
    print("Num nodes:", nodes_num)
    print("Num edges:", edges_num)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

    #import os
    #BASE_PATH = "/home/dblux/projects/phd/cs6216/graph_embedding/gae-master"
    #
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, os.path.join(BASE_PATH, "model.ckpt"))
    #print("Model saved in path: %s" % save_path)
    
#     Generate embeddings
#        emb = sess.run(model.embeddings, feed_dict=feed_dict)
#        print(emb.shape)

embedding_quality = pd.DataFrame({"id": file_list, "num_nodes": num_nodes_list, "num_edges": num_edges_list,
                                  "auc_score": auc_list, "ap_score": ap_list})

embedding_quality.to_csv("../../dump/embedding_16_do_0.1.tsv", sep="\t")
