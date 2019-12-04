import os, time, sys
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
import pandas as pd

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
file_index = int(sys.argv[1])

# Import gene expression data
expression_profile = pd.read_csv("../../data/raw/data_labelled.tsv", sep="\t")
# Convert row names from int to str
expression_profile.index = expression_profile.index.map(str)
id_list = list(expression_profile)
# List large pathway files
KEGG_DIR = "../../info/kegg_human-edgelist/large/"
fname = os.listdir(KEGG_DIR)

# Read graph as an undirected simple graph!
# Read as OrderedGraph if using python < 3.6
file = fname[file_index]
print(file)
fpath = os.path.join(KEGG_DIR, file)
G = nx.read_edgelist(fpath)
edges_num = len(G.edges())
# Creates ordered list of nodes
nodes = G.nodes()
nodes_num = len(nodes)

# Create adj matrix from graph
# adj is adjacency matrix: scipy sparse csr matrix
adj = nx.adjacency_matrix(G)

# Loop through all sample IDs for each pathway
for id in id_list: 
    # Fetch features using ordered nodes
    features_list = [expression_profile.loc[gene, id] for gene in nodes]
    # features is a scipy sparse lil matrix
    features = sp.lil_matrix(features_list)
    features = features.reshape(len(features_list),1)
  
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    
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
    
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    
    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]        
        
    # Generate embeddings
    embedding = sess.run(model.embeddings, feed_dict=feed_dict)
    output_fpath = "../../data/embedding/gae/{}-{}.npy".format(id, file[:-4])
    # Save embedding
    np.save(output_fpath, embedding)
    print(output_fpath)
    del model
        