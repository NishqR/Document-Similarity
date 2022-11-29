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

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('punkt')

#stop_words = stopwords.words('english')
stop_words = set(stopwords.words('english') + [word.strip("\n") for word in open("remove_words.txt", "r")])

lemmatizer = WordNetLemmatizer()


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

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

def process_model(articles_batch, embeddings_dict, count_dict, count):

    for article in articles_batch:
        
        print(f"Doing embeddings for article {count[0]}")
        
        start = time()
        
        sentences = sent_tokenize(article)
        
        for sentence in sentences:
            words_in_sentence = list(sentence.split(" "))
            for word_ in words_in_sentence:

                word_ = word_.lower()
                word_ = word_.strip()
                word_ = word_.replace(" ", "")
                word_ = word_.replace(",", "")
                word_ = word_.replace(".", "")
                word_ = word_.replace(":", "")

                if lemmatizer.lemmatize(word_) != 'ha' and lemmatizer.lemmatize(word_) != 'wa':
                    word_ = lemmatizer.lemmatize(word_)

                if word_ not in stop_words and word_ not in string.punctuation:
                    if word_ not in embeddings_dict.keys():
                        #embeddings_dict[word_] = model.encode(word_)
                        embeddings_dict[word_] = ""
                        count_dict[word_] = 1

                    else:
                        count_dict[word_] += 1

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))


def create_threads(articles_batch, embeddings_dict, count_dict, count):

    
    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    # This will download and load the pretrained model offered by UKPLab.
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    print('MODEL LOADED in %.2f seconds' % (time() - start))

    num_threads = 1
    threads_list = []

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
            thread = threading.Thread(target = process_model, 
                                        args = (articles_batch, embeddings_dict, count_dict, count))
            threads_list.append(thread)

if __name__ == "__main__":

    run_relevant = True
    main_start = time() 

    articles_df = pd.read_csv("all_articles.csv")

    num_runs = 0

    while(num_runs < 2):
    
        if run_relevant == True:

            articles = list(articles_df[articles_df['relevant_campbell'] == 1]['text'])

        else:

            articles = list(articles_df[articles_df['relevant_campbell'] == 0]['text'])        

        num_cpus = 3
        #num_cpus = multiprocessing.cpu_count()
        print(f"Processor count = {num_cpus}")
        
        num_threads = len(articles) / num_cpus
        print(f"Thread count = {num_threads}")

        processes_list = []
        print(f"Processes list = {processes_list}")

        start_index = 0

        embeddings_manager = multiprocessing.Manager()
        
        embeddings_dict = embeddings_manager.dict()
        count_dict = embeddings_manager.dict()

        count = embeddings_manager.list()
        count.append(0)
        
        for cpu_num in range(num_cpus):
            end_index = int((cpu_num+1)*(len(articles)/num_cpus))
            #print(end_index)
            #print(num_list[start_index:end_index])

            process = multiprocessing.Process(target = create_threads, args=(articles[start_index:end_index], embeddings_dict, count_dict, count))
            processes_list.append(process)
            start_index = int((cpu_num+1)*(len(articles)/num_cpus))

        print(f"Processes list = {processes_list}")
        
        for process in processes_list:
            print(f"Starting process")
            process.start()

        for process in processes_list:
            print(f"Double checking process")
            process.join()


        print(sorted(((v, k) for k, v in count_dict.items()), reverse=True))
        
        
        if run_relevant == True:
            relevant_df = pd.DataFrame(pd.Series(count_dict))
            relevant_df = relevant_df.sort_values(by=[0], ascending=False)
            relevant_df.to_csv("relevant_words_count.csv")
            
        else:
            irrelevant_df = pd.DataFrame(pd.Series(count_dict))
            irrelevant_df = irrelevant_df.sort_values(by=[0], ascending=False)
            irrelevant_df.sort_values(by=[0], ascending=False).to_csv("irrelevant_words_count.csv")
        
        num_runs += 1
        run_relevant = False

    relevant_words = list(relevant_df.head(20).index)
    irrelevant_words = list(irrelevant_df.head(20).index)
    common_words = intersection(relevant_words, irrelevant_words)
    
    print("COMMON WORDS")
    print(common_words)
    
    print("UNIQUE RELEVANT")
    print([x for x in relevant_words if x not in common_words])

    print("UNIQUE IRRELEVANT")
    print([x for x in irrelevant_words if x not in common_words])

    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))