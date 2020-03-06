# GCTM
This is an implementation of Graph Convolution Topic Model for Data Streams 
\
Data descriptions:
*  Training file, we used the bag of words format.
* Testing folder (E.g data).
* Graph, we saved by format: node1 \tab node2 \tab weight (E.g data/edgesw.txt). while node2 has a relationship (E.g synonym or antonym ) with node1 and the weight of each edge is computed by Wu-Palmer or Cosine similarity.

\
Requirements:
* python 3.7.
* pytorch 1.2.0

\
Run the demo:
* python runGCTM.py
