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
from statistics import mean

import gensim
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer

from tensorflow_estimator.python.estimator.canned.dnn import dnn_logit_fn_builder
import tensorflow_hub as hub
import tensorflow as tf

import multiprocessing
from multiprocessing import Pool, cpu_count
import threading

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize

stop_words = stopwords.words('english')

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def process_bert_similarity(base_embeddings, documents):
    start = time()

    scores = cosine_similarity([base_embeddings], [documents]).flatten()

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    if highest_score > 0.95:
        highest_score = 0

    print('Cell took %.2f seconds to run.' % (time() - start))
    return highest_score

def process_model(articles_batch, all_articles, matrices_dict, cpu_num, count):
    for article_comp in articles_batch:

        temp_list = []
        
        for article_against in all_articles:

            score = process_bert_similarity(article_comp, article_against)
            count[0] += 1
            print(f"Count - {count[0]}/16129")
            temp_list.append(score)

        matrices_dict[cpu_num].append(list(temp_list))
        temp_list = []


def create_threads(articles_batch, all_articles, matrices_dict, cpu_num, count):

    num_threads = 1
    threads_list = []

    list_manager = multiprocessing.Manager()
    temp_list = list_manager.list()

    for r in range(num_threads + 1):

        if len(threads_list) == num_threads:

            # Start the processes       
            for thread in threads_list:
                #print(f"Starting threads")
                thread.start()

            # Ensure all of the processes have finished
            for thread in threads_list:
                #print(f"Double checking threads")
                thread.join()
                #temp_matrix.append(list(temp_list))

            threads_list = []


        else:
            thread = threading.Thread(target = process_model, 
                                        args = (articles_batch, all_articles, matrices_dict, cpu_num, count))
            threads_list.append(thread)

if __name__ == "__main__":

    main_start = time() 
    
    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    # This will download and load the pretrained model offered by UKPLab.
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    stop_words = set(stopwords.words('english'))
    print('MODEL LOADED in %.2f seconds' % (time() - start))

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    lemmatizer = WordNetLemmatizer()

    articles_df = pd.read_csv("all_articles.csv")

    #articles = list(articles_df['text'])
    articles = []

    embeddings_start = time()

    embeddings_dict = {}
    count_dict = {}
    for index, row in articles_df.iterrows():
        if index < 25:

            #print(f"Doing embeddings for article {index}")
            start = time()
            
            print(row['text'])
            print(len(row['text']))
            print("--------------------------------------------------------------------------------------------")
            sentences = sent_tokenize(row['text'])
            print(sentences)
            print(len(sentences))
            print("--------------------------------------------------------------------------------------------")
            #print(sentences[1])
            #print(sentences[1].split(" ")[0])

            for sentence in sentences:
                words_in_sentence = list(sentence.split(" "))
                for word_ in words_in_sentence:

                    word_ = word_.lower()
                    word_ = word_.strip()
                    word_ = word_.replace(" ", "")
                    word_ = lemmatizer.lemmatize(word_)

                    word_embedding = model.encode(word_)
                    if word_ not in stop_words and word_ not in string.punctuation:
                        if word_ not in embeddings_dict.keys():
                            embeddings_dict[word_] = model.encode(word_)
                            count_dict[word_] = 1

                        else:
                            count_dict[word_] += 1

            print(sorted(((v, k) for k, v in count_dict.items()), reverse=True))

            
            '''embedding_1 = model.encode("finance")
            embedding_2 = model.encode("financial")
            embedding_3 = model.encode("corporation")
            embedding_4 = model.encode("money")

            lemmatized = lemmatizer.lemmatize("corpora")
            print(lemmatized)
            '''
            # lowercase
            # remove trailing spaces
            # lemmatize
            # add count and embedding to dict

            #print(cosine_similarity([embedding_1], [embedding_2]).flatten())
            #print(cosine_similarity([embedding_1], [embedding_2]).flatten())
            #print(cosine_similarity([embedding_1], [embedding_3]).flatten())
            #print(cosine_similarity([embedding_1], [embedding_4]).flatten())
            #print(embedding_1, embedding_2, embedding_3, embedding_4)
            '''
            base_embeddings_sentences = model.encode(sentences)
            print(base_embeddings_sentences)
            print(len(base_embeddings_sentences[1]))
            print("--------------------------------------------------------------------------------------------")
            base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
            print(np.array(base_embeddings_sentences))
            print(len(np.array(base_embeddings_sentences)[1]))
            print("--------------------------------------------------------------------------------------------")
            articles.append(base_embeddings)
            print('Cell took %.2f seconds to run.' % (time() - start))
    
            '''
    print('All emeddings completed in %.2f seconds' % (time() - embeddings_start))

    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))