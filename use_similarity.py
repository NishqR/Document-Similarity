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

def process_use_similarity(base_embeddings, embeddings):
    start = time()
    scores = cosine_similarity(base_embeddings, embeddings).flatten()

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    #most_similar_document = documents[highest_score_index]
    #print("Most similar document by USE with the score:", most_similar_document, highest_score)
    if highest_score > 0.95:
        highest_score = 0

    print('Cell took %.2f seconds to run.' % (time() - start))
    return highest_score

def process_model(articles_batch, all_articles, matrices_dict, cpu_num, count):
    for article_comp in articles_batch:

        temp_list = []
        #article_comp = preprocess()
        base_embeddings = model([article_comp])

        for article_against in all_articles:

            embeddings = model([article_against])

            score = process_use_similarity(base_embeddings, embeddings)
            count[0] += 1
            print(f"Count - {count[0]}/16129")
            temp_list.append(score)

        matrices_dict[cpu_num].append(list(temp_list))
        temp_list = []


def create_threads(articles_batch, all_articles, matrices_dict, cpu_num, count):

    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    filename = "use/universal-sentence-encoder_4"
    model = hub.load(filename)
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
    
    num_cpus = 6
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


    matrix = []
    for index, row in articles_df.iterrows():
        
        temp_scores_list = []
        
        if row['relevant_campbell'] == 1:
            
            for index, row in articles_df.iterrows():
                
                if row['relevant_campbell'] == 1:
                    
                    temp_scores_list.append(1)
                
                else:
                    
                    temp_scores_list.append(0)
                    
        else:
            
            temp_scores_list = list(np.zeros(len(articles_df)))
            
        
        matrix.append(temp_scores_list)
        

    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(matrix, fignum=1)
    plt.savefig('base_matrix.png')
    #plt.show()
    
    temp_matrix = []
    for cpu_num in range(num_cpus):
        for temp_list in matrices_dict[cpu_num]:
            temp_matrix.append(temp_list)


    print(temp_matrix)
    print(f"LENGTH OF TEMP MATRIX: {len(temp_matrix)}")
    #print(temp_matrix)

    max_similarity = 0
    min_similarity = 10000

    for list_ in temp_matrix:
        for val_ in list_:

            if val_ > max_similarity:
                max_similarity = val_

            if val_ < min_similarity:
                min_similarity = val_

    wmd_matrix = []

    for temp_scores_list in temp_matrix:

        normalized_list = []

        for similarity_value in temp_scores_list:

            normalized_similarity = similarity_value / max_similarity
            normalized_list.append(normalized_similarity)

        wmd_matrix.append(normalized_list)

    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(wmd_matrix, fignum=1)
    plt.savefig('results/use/use.png')

    wmd_diff_matrix = []
    for i in range(len(matrix)):
        
        temp_list = []
        for j in range(len(matrix[i])):
            temp_list.append(matrix[i][j] - wmd_matrix[i][j])
        
        wmd_diff_matrix.append(temp_list)
        
    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(wmd_diff_matrix, fignum=1)

    plt.savefig('results/use/use_diff.png')

        # EVALUATION SUITE
    relevance_threshold = 0.6
    true_positive = 0
    false_negative = 0
    true_negative = 0
    false_positive = 0

    total_relevant = 0
    total_irrelevant = 0
    for i in range(len(matrix)):
        
        for j in range(len(matrix[i])):

            if matrix[i][j] == 1:

                total_relevant += 1

                if wmd_matrix[i][j] > relevance_threshold:
                    true_positive += 1

                if wmd_matrix[i][j] < relevance_threshold:
                    false_negative += 1

            if matrix[i][j] == 0:

                total_irrelevant += 1

                if wmd_matrix[i][j] > relevance_threshold:
                    false_positive += 1

                if wmd_matrix[i][j] < relevance_threshold:
                    true_negative += 1

    eval_dict = {}

    print(f"Total relevant = {total_relevant}")
    print(f"Total total_irrelevant = {total_irrelevant}")

    print(f"True positive = {true_positive}")
    print(f"True Negative = {true_negative}")
    print(f"False_negative = {false_negative}")
    print(f"false_positive = {false_positive}")

    eval_dict['true_positive'] = true_positive
    eval_dict['true_negative'] = true_negative
    eval_dict['false_positive'] = false_positive
    eval_dict['false_negative'] = false_negative
    eval_dict['accuracy'] = (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
    eval_dict['precision'] = true_positive / (true_positive + false_positive)
    eval_dict['recall'] = true_positive / (true_positive + false_negative)
    eval_dict['f1_score'] = (2 * eval_dict['precision'] * eval_dict['recall']) / (eval_dict['precision'] + eval_dict['recall']) 
    #print(f"correctly_classified documents  = {correctly_classified}")
    print(f"Accuracy = {eval_dict['accuracy']}")
    print(f"Precision = {eval_dict['precision']}")
    print(f"Recall = {eval_dict['recall']}")
    print(f"F1-Score = {eval_dict['f1_score']}")

    eval_df = pd.DataFrame (pd.Series(eval_dict)).T
    eval_df.to_csv('results/use/use_metrics.csv')

    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))