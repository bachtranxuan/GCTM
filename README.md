




# GCTM
This is an implementation of [Graph Convolution Topic Model for Data Streams](link).

Some benefits of our model
*	 GCTM exploits a knowledge graph, which comes from human knowledge or a pre-trained model, to enrich a topic model for data streams, especially in case of sparse or noisy data. We emphasize that our work first provides a way to model prior knowledge of graph form in a streaming environment
*	We also propose an automatic mechanism to balance between the original prior knowledge and old knowledge learnt in the previous minibatch. This mechanism can automatically control the impact of the prior knowledge in each minibatch. When concept drift happens, it can automatically decrease the influence of the old knowledge but increase the influence of the prior knowledge to help our method deal well with the concept drift.

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
*	Training file, we used the bag of words format. (E.g data/train.txt)
```
		4 14:1 12:2 7:2 96:1
```
*	Testing folder, including one or more pair file (part_1, part_2). Each document in the test set is divided randomly into two disjoint part ![formula](https://render.githubusercontent.com/render/math?math=w_{obs}) (part_1) and ![formula](https://render.githubusercontent.com/render/math?math=w_{ho}) (part_2) with a ratio of 4:1. (E.g data).
*	Graph, we saved by format: node1 \tab node2 \tab weight (E.g data/edgesw.txt). while node2 has a relationship (E.g synonym or antonym ) with node1 and the weight of each edge is computed by Wu-Palmer (when using [WordNet](https://wordnet.princeton.edu/)) or Cosine similarity (when using a pre-trained model on a big dataset such as [Word2vec](https://nlp.stanford.edu/projects/glove/)).
```
		0	0	1
		0	20	0.8
		1	1	1
		1	2	0.2
```

## Result
We compare our model with three state-of-the-art base-lines:
SVB ([Broderick et al., 2013](https://arxiv.org/pdf/1307.6769.pdf)), PVB ([McInerney et al.,  2015](https://arxiv.org/pdf/1507.05253.pdf)) and SVP-PP ([Masegosa et al., 2017](http://proceedings.mlr.press/v70/masegosa17a/masegosa17a.pdf)). Log predictive probability ([LPP](http://jmlr.org/papers/v14/hoffman13a.html))  and Normalized pointwise mutual information ([NPMI](https://www.aclweb.org/anthology/E14-1056/))
![Log predictive probability](./figures/perplexities.png)
![Normalized pointwise mutual information](./figures/npmi.png)

## Citation
if you find that TPS is useful for your research, please citing:
```
@article{*,
  title={Graph Convolution Topic Model for Data Streams},
  author={*},
  journal={arXiv preprint arXiv},
  year={2020}
}
```
