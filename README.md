




# GCTM
This is an implementation of Graph Convolution Topic Model for Data Streams

## Installation
1. Clone the repository
```
		https://github.com/bachtranxuan/GCTM.git
``` 
2. Requirements environment
```
		Python 3.7
		Pytorch 1.2.0
		Numpy, Scipy
```
## Training
You can run with command
```
	python runGCTM.py
```
## Data descriptions
*	Training file, we used the bag of words format.
*	Testing folder, including one or more pair file (part_1, part_2). Each document in the test set is divided randomly into two disjoint part ![formula](https://render.githubusercontent.com/render/math?math=w_{obs}) (part_1) and ![formula](https://render.githubusercontent.com/render/math?math=w_{ho}) (part_2) with a ratio of 4:1. (E.g data).
*	Graph, we saved by format: node1 \tab node2 \tab weight (E.g data/edgesw.txt). while node2 has a relationship (E.g synonym or antonym ) with node1 and the weight of each edge is computed by Wu-Palmer (when using [WordNet](https://wordnet.princeton.edu/)) or Cosine similarity (when using a pre-trained model on a big dataset such as [Word2vec](https://nlp.stanford.edu/projects/glove/)).
## Model
Graphical representation for GCTM is illustrated below
\
<img src="figures/model.png" alt="centered image" height="450px" width="500px" >
## Result
We compare our model with three state-of-the-art base-lines:
SVB ([Broderick et al., 2013](https://arxiv.org/pdf/1307.6769.pdf)), PVB ([McInerney et al.,  2015](https://arxiv.org/pdf/1507.05253.pdf)) and SVP-PP ([Masegosa et al., 2017](http://proceedings.mlr.press/v70/masegosa17a/masegosa17a.pdf)). Log predictive probability ([LPP](http://jmlr.org/papers/v14/hoffman13a.html))  and Normalized pointwise mutual information ([NPMI](https://www.aclweb.org/anthology/E14-1056/))
![Log predictive probability](./figures/perplexities.png)
![Normalized pointwise mutual information](./figures/npmi.png)
``` 
