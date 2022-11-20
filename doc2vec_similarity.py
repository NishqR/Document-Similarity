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
lemmatizer = WordNetLemmatizer()

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess(text):
    # Steps:
    # 1. lowercase
    # 2. Lammetize. (It does not stem. Try to preserve structure not to overwrap with potential acronym).
    # 3. Remove stop words.
    # 4. Remove punctuations.
    # 5. Remove character with the length size of 1.

    lowered = str.lower(text)

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(lowered)

    words = []
    for w in word_tokens:
        if w not in stop_words:
            if w not in string.punctuation:
                if len(w) > 1:
                    lemmatized = lemmatizer.lemmatize(w)
                    words.append(lemmatized)

    return words

def process_doc2vec_similarity(base_vector, vectors):

    start = time()
    # Both pretrained models are publicly available at public repo of jhlau.
    # URL: https://github.com/jhlau/doc2vec
    
    scores = cosine_similarity([base_vector], [vectors]).flatten()

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

def process_embeddings(articles_batch, articles_dict, cpu_num, count):
    for article in articles_batch:

        print(f"Doing embeddings for article {count[0]}")
        start = time()
        tokens = preprocess(article)
        # Only handle words that appear in the doc2vec pretrained vectors. enwiki_ebow model contains 669549 vocabulary size.
        tokens = list(filter(lambda x: x in model.wv.vocab.keys(), tokens))
        base_vector = model.infer_vector(tokens)
        #print("BASE VECTOR TYPE")
        #print(type(base_vector))
        #print(base_vector)
        articles_dict[cpu_num].append(base_vector)
        #print(articles_dict[cpu_num])
        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))
        

def create_threads(articles_batch, articles_dict, cpu_num, count):

    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    filename = 'doc2vec/doc2vec.bin'
    model= Doc2Vec.load(filename)
    print('MODEL LOADED in %.2f seconds' % (time() - start))

    num_threads = 1
    threads_list = []

    #list_manager = multiprocessing.Manager()
    #temp_list = list_manager.list()

    for r in range(num_threads + 1):

        if len(threads_list) == num_threads:

            # Start the processes       
            for thread in threads_list:
                thread.start()

            # Ensure all of the processes have finished
            for thread in threads_list:
                thread.join()

            threads_list = []

        else:
            thread = threading.Thread(target = process_embeddings, 
                                        args = (articles_batch, articles_dict, cpu_num, count))
            threads_list.append(thread)

if __name__ == "__main__":

    main_start = time() 
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    nltk.download('punkt')


    articles_df = pd.read_csv("all_articles.csv")
    articles = list(articles_df['text'])

    num_cpus = 5
    #num_cpus = multiprocessing.cpu_count()
    print(f"Processor count = {num_cpus}")
    
    num_threads = len(articles) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    print(f"Processes list = {processes_list}")

    start_index = 0

    articles_manager = multiprocessing.Manager()
    articles_dict = articles_manager.dict()

    for i in range(num_cpus):
        articles_dict[i] = articles_manager.list()
    
    count = articles_manager.list()
    count.append(0)

    for cpu_num in range(num_cpus):
        end_index = int((cpu_num+1)*(len(articles)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_threads, args=(articles[start_index:end_index], articles_dict, cpu_num, count))
        processes_list.append(process)
        start_index = int((cpu_num+1)*(len(articles)/num_cpus))

    print(f"Processes list = {processes_list}")
    
    for process in processes_list:
        print(f"Starting process")
        process.start()

    for process in processes_list:
        print(f"Double checking process")
        process.join()

    print("ALL ARTICLES TOKENIZED")


    tokenized_articles = []
    for cpu_num in range(num_cpus):
        #for temp_list in articles_dict[cpu_num]:
        for article in articles_dict[cpu_num]:
            tokenized_articles.append(article)

    print(f"Length of tokenized articles list = {len(tokenized_articles)}")

    tokenized_df = pd.DataFrame (tokenized_articles)
    tokenized_df.to_csv('doc2vec_tokenized_articles.csv')

    print("Tokenized articles saved to csv")
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



    temp_list = []
    temp_matrix = []

    min_similarity = 10000
    max_similarity = 0
    for article_base in tokenized_articles:
    
        temp_list = []
        
        for article_other in tokenized_articles:
            
            start = time()
            
            similarity = process_doc2vec_similarity(article_base, article_other)
            
            print('Cell took %.2f seconds to run.' % (time() - start))
            

            if similarity > max_similarity:
                max_similarity = similarity
                
            if similarity < min_similarity:
                min_similarity = similarity 
                
            temp_list.append(similarity)
                    
        temp_matrix.append(temp_list)


    doc2vec_matrix = [] 

    for temp_scores_list in temp_matrix:

        normalized_list = []

        for similarity_value in temp_scores_list:

            normalized_similarity = similarity_value / max_similarity
            normalized_list.append(normalized_similarity)

        doc2vec_matrix.append(normalized_list)

    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(doc2vec_matrix, fignum=1)
    plt.savefig('results/doc2vec/doc2vec.png')

    doc2vec_diff_matrix = []
    for i in range(len(matrix)):
        
        temp_list = []
        for j in range(len(matrix[i])):
            temp_list.append(matrix[i][j] - doc2vec_matrix[i][j])
        
        doc2vec_diff_matrix.append(temp_list)
        
    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(doc2vec_diff_matrix, fignum=1)

    plt.savefig('results/doc2vec/doc2vec_diff.png')

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

                if doc2vec_matrix[i][j] > relevance_threshold:
                    true_positive += 1

                if doc2vec_matrix[i][j] < relevance_threshold:
                    false_negative += 1

            if matrix[i][j] == 0:

                total_irrelevant += 1

                if doc2vec_matrix[i][j] > relevance_threshold:
                    false_positive += 1

                if doc2vec_matrix[i][j] < relevance_threshold:
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
    eval_df.to_csv('results/doc2vec/doc2vec_metrics.csv')
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))