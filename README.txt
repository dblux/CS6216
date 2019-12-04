CS6216 Project: Learning Graph Representations for Use in Cancer Metastasis Prediction
Author: Chan Wei Xin

===============
R dependencies
===============
KEGGgraph
Rgraphviz
igraph

===============
Python dependencies
===============
networkx==2.2
scipy==1.1.0
setuptools==40.6.3
numpy==1.15.4
tensorflow==1.13.1
keras==2.2.2
matplotlib--2.2.3

R scripts were used to pre-process pathway data:
- preprocess_kegg.R
- preprocess_entrez.R
- preprocess_indv_gene.R
- generate_genelist.R

To generate graph representations using GAE (python):
1. Change directory to graph_embedding/gae-master
2. ./run_gae.sh 0 157

To perform the classification task use jupyter notebooks (python):
- models.ipynb
- keras.ipynb

To visualise the results (python):
- visualisation.py
- visualisation.ipynb

Due to size constraints only crucial data files are provided!
