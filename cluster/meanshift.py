############
## IMPORT ##
############

import os
import sys

# local modules in src
directory_to_prepend = os.path.abspath("src")
if directory_to_prepend not in sys.path:
    sys.path = [directory_to_prepend] + sys.path

import numpy as np
import pandas as pd

from sklearn.cluster import MeanShift
from joblib import load

import matplotlib.pyplot as plt
from src.plot import plot_2d
from src.utils import convert_string_to_float_array

from sklearn.decomposition import PCA
pca_2d = PCA(n_components=2)

# path_to_dataset_with_text =
# path_to_dataset_without_text =
# path_to_meanshift_model =

########################################
## CREATE WORD,TEXT,CLUSTER CSVs ##
########################################

#
# df = pd.read_csv(path_to_dataset_with_text)
# print(f"-> {df.columns}")
# print(f"-> with dtypes: {df.dtypes}")
#
# keys = ['combined_sentence', 'text_vec', 'weighted01', 'weighted02', 'weighted03', 'weighted04', 'weighted05', 'weighted06', 'weighted07', 'weighted08', 'weighted09', 'word_vec']
#
# print("-> csv with word,text,cluster_label is being created")
# for key in keys:
    # df[['word', 'text', key + '_cluster']].sort_values(by=[key + '_cluster']).to_csv('./processed_words_with_text_' + key + '_clusters.csv')
    # print(f"-> for {key} csv is generated")
#
# print("-> Generating one general csv with all cluster labels and word/text pair")
#
# _clusters = []
# for key in keys:
    # _clusters.append(key + '_cluster')
#
# df[['word', 'text', *_clusters]].sort_values(by=['word']).to_csv('./processed_word_with_text_all_clusters.csv')
# print("-> csv with word,text,cluster labels for all methods is generated ")










################################ unknown universe














# save csv file word,text,combined_sentence_cluster sorted by combined_sentence_cluster

# array of vectors for each row in the dataset
# combined_sentences = convert_string_to_float_array(df, 'combined_sentence')

# print(f"combined sentences has shape: {combined_sentences.shape}")
# print(f"combined sentences has type: {type(combined_sentences)}")
# print(f"combined sentences[i] data has shape: {(combined_sentences[0]).shape}")
# print(f"combined sentences[i] data has type: {type(combined_sentences[0])}")
# print(f"combined sentences[i,i] data has shape: {(combined_sentences[0][0]).shape}")
# print(f"combined sentences[i,i] data has type: {type(combined_sentences[0][0])}")

# embeddings_2d = pca_2d.fit_transform(combined_sentences)
# plot_2d(embeddings_2d[:,0], embeddings_2d[:,1], clusters=df['combined_sentence_cluster'], save='TEST.pdf')

# print(f"example read combined sentence vector type: {(df.loc[0,'combined_sentence'])}")

# plt.scatter(0)
# plot_2d(df.loc[:,], df[:,1], save='fasttext_preprocessed_word_data_without_text_' + 'word_vec' + '_vectors_2d.pdf')

# TEXT VEC

# save csv file word,text,text_vec_cluster sorted by combined_sentence_cluster
