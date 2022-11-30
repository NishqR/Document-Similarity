import os
import string
from time import time
import random

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from difflib import SequenceMatcher
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from statistics import mean

import gensim
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer


print("---------------------------- LOADING MODEL---------------------------- ")
start = time()
global model
# This will download and load the pretrained model offered by UKPLab.
model = SentenceTransformer('bert-base-nli-mean-tokens')

print('MODEL LOADED in %.2f seconds' % (time() - start))

embedding_1 = model.encode("cent")
#print(embedding_1)
embedding_2 = model.encode("cent")
embedding_3 = model.encode("cent")
embedding_4 = model.encode("cent")

# lowercase
# remove trailing spaces
# lemmatize
# add count and embedding to dict

print(cosine_similarity([embedding_1], [embedding_2]).flatten())
print(cosine_similarity([embedding_1], [embedding_3]).flatten())
print(cosine_similarity([embedding_1], [embedding_4]).flatten())
#print(embedding_1, embedding_2, embedding_3, embedding_4)
'''
base_embeddings_sentences = model.encode(sentences)
base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
articles.append(base_embeddings)
'''