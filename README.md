# Project Structure

```
.
├── data
│   ├── english_words.csv (word-text dataset)
│   └── ...
│
├── models
│   ├── cc.en.300.bin (fasttext) 
│   └── ...
│
├── fastext.ipynb (embedding creation)
│ 
├── src
│   ├── plot.py
│   ├── preprocess.py
│   └── utils.py
│ 
├── environment.yml│
└── README.md (me)
```


DISTANCE-BASED:
- K-means: distance based, look at heuristics for choosing K (elbow, gap statistic, silhouette and do not forget cross validation)

DISTRIBUTION BASED:
- Gaussian mixture model (GMM): with assumption cluster prior is categorical and each cluster has multivariate gaussian, learning with expectation-maximization (EM) algorithm

HIERARCHICAL BASED:
- agglomerative clustering
- divise clustering

MINIMUM CUT BASED:
- spectral clustering: finds minimum cut

PROBABILISTIC BASED (with uncertainty) can handle noisy data and also generate new data (link prediction):
- planted partition model (PPM)
- stochastic block model (SBM)

NN BASED:
- AE
- deep embedding clustering

-----------------------------------------

Software used:
- scikit-learn
- fasttext
- pandas
- numpy
- jupyterlab
- matplotlib
