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

def process_wmd_similarity(article_comp, article_against):
    start = time()
    #base_document = preprocess(base_document)
    #documents = preprocess(documents[0])
    
    distance = model.wmdistance(article_comp, article_against)
    #distance = model
    #print(f"Distance = {distance}")
    print('Cell took %.2f seconds to run.' % (time() - start))
    
    try: 
        score = 1 / distance
    except:
        score = 0


    return score

def process_model(articles_batch, all_articles, matrices_dict, cpu_num, count):
    for article_comp in articles_batch:

        temp_list = []
        article_comp = preprocess(article_comp)

        for article_against in all_articles:

            article_against = preprocess(article_against)

            score = process_wmd_similarity(article_comp, article_against)
            count[0] += 1
            print(f"Count - {count[0]}/16129")
            temp_list.append(score)

        matrices_dict[cpu_num].append(list(temp_list))
        temp_list = []


def create_threads(articles_batch, all_articles, matrices_dict, cpu_num, count):

    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    #model = 10000
    model = gensim.models.KeyedVectors.load_word2vec_format('wmd/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('MODEL LOADED in %.2f seconds' % (time() - start))

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
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    lemmatizer = WordNetLemmatizer()

    articles_df = pd.read_csv("all_articles.csv")

    articles = list(articles_df['text'])
    
    num_cpus = 5
    #num_cpus = multiprocessing.cpu_count()
    print(f"Processor count = {num_cpus}")
    
    num_threads = len(articles_df) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    print(f"Processes list = {processes_list}")

    start_index = 0

    matrix_manager = multiprocessing.Manager()
    matrices_dict = matrix_manager.dict()

    for i in range(num_cpus):
        matrices_dict[i] = matrix_manager.list()
    #temp_matrix = matrix_manager.list()

    count = matrix_manager.list()
    count.append(0)

    for cpu_num in range(num_cpus):
        end_index = int((cpu_num+1)*(len(articles_df)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_threads, args=(articles[start_index:end_index], articles, matrices_dict, cpu_num, count))
        processes_list.append(process)
        start_index = int((cpu_num+1)*(len(articles_df)/num_cpus))

    print(f"Processes list = {processes_list}")
    
    for process in processes_list:
        print(f"Starting process")
        process.start()

    for process in processes_list:
        print(f"Double checking process")
        process.join()
    

    temp_matrix = []
    for cpu_num in range(num_cpus):
        for temp_list in matrices_dict[cpu_num]:
            temp_matrix.append(temp_list)


    np.save('results/wmd/wmd_matrix.npy', np.array(temp_matrix))
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))