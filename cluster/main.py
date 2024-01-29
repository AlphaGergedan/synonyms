############
## IMPORT ##
############

import os
import sys

# local modules in src
directory_to_prepend = os.path.abspath("src")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

# import local modules here
from preprocess import preprocess, get_combined_sentence_embedding, get_weighted_embedding, split_with_without_text
from utils import is_valid_text

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
#MeanShift, AgglomerativeClustering, OPTICS, Birch, MiniBatchKMeans
#from sklearn.mixture import GaussianMixture
# mean-shift
# ward-hierarchical clustering
# aggloromative clustering
# OPTICS
# Gaussian Mixture
# BIRCH

import fasttext as ft
import fasttext.util as futil
#futil.download_model('de', if_exists='ignore')
path_to_fasttext_model = './models/cc.de.300.bin'
assert os.path.exists(path_to_fasttext_model)
ft_model = ft.load_model(path_to_fasttext_model)

# for plotting
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca_2d = PCA(n_components=2)
from plot import plot_2d, plot_parameter_search

# for saving clustering models
import joblib

# COMMON PARAMS
random_seed = 12345

print('-> Imports completed.')

############################
## DATASET AND PREPROCESS ##
############################

# original dataset
path_to_dataset = '../data/path-to-dataset.csv'
assert os.path.exists(path_to_dataset)

# process text (append)
df = pd.read_csv(path_to_dataset, usecols=['word', 'text'])
print(f'-> Dataset is read, of shape {df.shape}')

df = preprocess(df, remove_duplicates=False)
print(f"-> Preprocess is done.")

################
## EMBEDDINGS ##
################

# LIST YOUR EMBEDDING HERE TO CREATE CLUSTERS ACCORDING TO IT
embeddings = {
    # appends word to the text and uses sentence embedding
    'combined_sentence': (lambda row: get_combined_sentence_embedding(ft_model, row)),
    # uses word embedding for word and sentence embedding for the text (weighted)
    'text_vec': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0)),
    'weighted01': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.1)),
    'weighted02': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.2)),
    'weighted03': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.3)),
    'weighted04': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.4)),
    'weighted05': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.5)),
    'weighted06': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.6)),
    'weighted07': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.7)),
    'weighted08': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.8)),
    'weighted09': (lambda row: get_weighted_embedding(ft_model, row, word_weight=0.9)),
    'word_vec': (lambda row: get_weighted_embedding(ft_model, row, word_weight=1)),
}

# split dataset into rows with text and without text
# TODO double check
num_words_with_text = 0
num_words_without_text = 0
for index, row in df.iterrows():
    if type(row['text']) == str and len(row['text']) > 0:
        num_words_with_text = num_words_with_text + 1
    else:
        num_words_without_text = num_words_without_text + 1

df_with_text, df_without_text = split_with_without_text(df)

assert num_words_without_text == df_without_text.shape[0]
assert num_words_with_text == df_with_text.shape[0]

print(f'-> #words with text: {df_with_text.shape[0]}')
print(f'-> #words without text: {df_without_text.shape[0]}')

# create embeddings for the words with text
print(f"-> Creating embeddings for words with text...")
for key in embeddings:
    df_with_text[key] = df.apply(lambda row: embeddings[key](row), axis=1)
    print(f"-> Embedding using method '{key}' is created.")

print(f"-> Creating embeddings (word vectors) for words without text...")
df_without_text['word_vec'] = df.apply(lambda row: embeddings['word_vec'](row), axis=1)
print(f"-> Embedding using method 'word_vec' is created.")

# plot the results
for key in embeddings:
    # save the 2d plot to visualize
    # rows with text should be clustered in the end
    embeddings_2d = pca_2d.fit_transform(df_with_text[key].values.tolist())
    plot_2d(embeddings_2d[:,0], embeddings_2d[:,1], save='fasttext_preprocessed_words_data_with_text_' + key + '_vectors_2d.pdf')

# rows without text should be assigned to synonym clusters (final goal)
# all embeddings leads to the word embedding, therefore there is only one embedding available for the words without text (word vector)
embeddings_2d = pca_2d.fit_transform(df_without_text['word_vec'].values.tolist())
plot_2d(embeddings_2d[:,0], embeddings_2d[:,1], save='fasttext_preprocessed_words_data_without_text_' + 'word_vec' + '_vectors_2d.pdf')

print(f'-> Processed dataset vectors (with text and without text) 2D plot added under plots.')

###############
## MeanShift ##
###############

# we will have MeanShift clusters for each embedding of the words with text, number of clusters are set automatically
meanshift_clusters = {}

print('-> Initializing MeanShift clusters...')
for key in embeddings:
    meanshift_clusters[key] = MeanShift().fit(df_with_text[key].values.tolist())
    # save
    joblib.dump(meanshift_clusters[key], 'meanshift_model_on_' + key + '.pkl')

    n_clusters = len(np.unique(meanshift_clusters[key].labels_))
    print(f"-> MeanShift cluster created using embeddings from '{key}'. #clusters is {n_clusters}")

    print(f"-> Assigning words with texts...")
    df_with_text[key + '_cluster'] = meanshift_clusters[key].predict(pd.DataFrame(df_with_text[key].values.tolist()))
    print(f"-> Words with texts are assigned: {df_with_text.shape}")

    print(f"-> Assigning words without texts...")
    df_without_text[key + '_cluster'] = meanshift_clusters[key].predict(pd.DataFrame(df_without_text['word_vec'].values.tolist()))
    print(f"-> Words without texts are assigned: {df_without_text.shape}")

# save the df as csv
df_with_text.to_csv('./processed_words_with_text.csv')
df_without_text.to_csv('./processed_words_without_text.csv')

# TODO also plot the clusters on 2d
#plt.figure(1)
#plt.clf()

#colors = ["#dede00", "#377eb8", "#f781bf"]
#markers = ["x", "o", "^"]

#for k, col in zip(range(n_clusters_), colors):
#    my_members = labels == k
#    cluster_center = cluster_centers[k]
#    plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
#    plt.plot(
#        cluster_center[0],
#        cluster_center[1],
#        markers[k],
#        markerfacecolor=col,
#        markeredgecolor="k",
#        markersize=14,
#    )
#plt.title("Estimated number of clusters: %d" % n_clusters_)
#plt.show()

assert False

#############################################################################################
# TODO
# run k-means on the vectors with elbow method to define the number of k for the clustering
sum_of_squared_distances = []
K = range(1,num_words_with_text)
for k in K:
    km = KMeans(init='‘k-means++', random_state=random_seed, n_init='auto', algorithm='lloyd', n_clusters=k)
    km = km.fit(pd.DataFrame(df['embedding'].values.tolist()))
    sum_of_squared_distances.append(km.inertia_)

# plot it
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel(r'$k$')
plt.ylabel('sum of squared distances')
plt.title(r'Elbow Method For Optimal $k$')
plt.savefig("./k_means_clustering_distances.pdf")

# FIRST-STEP:   K = num_words_with_text and run K-Means
# SECOND-STEP:



#################################################################
## MeanShift + Remove Outliers with Text + K-Means on the Rest ##
#################################################################



#####################
## With Classifier ##
#####################
