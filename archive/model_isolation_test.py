import os
import string
import time
#from time import time

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from difflib import SequenceMatcher
#from collections import defaultdict
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#from statistics import mean

#import gensim
#from gensim.models.doc2vec import Doc2Vec
#from sentence_transformers import SentenceTransformer

#from tensorflow_estimator.python.estimator.canned.dnn import dnn_logit_fn_builder
#import tensorflow_hub as hub
#import tensorflow as tf

import multiprocessing
from multiprocessing import Pool, cpu_count

#import nltk
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#from nltk import sent_tokenize

import threading
import random

stop_words = stopwords.words('english')

def preprocess(sentence):

    sentence = sentence.lower().split()
    return [w for w in sentence.lower().split() if w not in stop_words]

'''
def process_model(dummy, x):
    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'
    sentence_obama = sentence_obama.lower().split()
    sentence_president = sentence_president.lower().split()

    # Remove stopwords.
    stop_words = stopwords.words('english')
    sentence_obama = [w for w in sentence_obama if w not in stop_words]
    sentence_president = [w for w in sentence_president if w not in stop_words]

    distance = model.wmdistance(sentence_obama, sentence_president)
    print(f"Distance = {distance}")

'''

def process_model(dummy, temp_list):

    out_list = []
    print(f"Model = {model}")
    for i in range(model):
        rando_number = random.random()
        #print(rando_number)
        temp_list.append(rando_number)

    print(len(temp_list))

def create_threads(num_cpus, matrices_list):

    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time.time()
    global model
    model = 10
    #model = gensim.models.KeyedVectors.load_word2vec_format('wmd/GoogleNews-vectors-negative300.bin.gz', binary=True)
    print('Cell took %.2f seconds to run.' % (time.time() - start))
    
    num_threads = 1
    threads_list = []

    list_manager = multiprocessing.Manager()
    temp_list = list_manager.list()

    for r in range(num_threads + 1):
        #print(f"Jobs list = {jobs_list}")

        if len(threads_list) == num_threads:

            #print(f"Threads list full")

            # Start the processes       
            for thread in threads_list:
                #print(f"Starting threads")
                thread.start()

            # Ensure all of the processes have finished
            for thread in threads_list:
                #print(f"Double checking threads")
                thread.join()
                matrices_list.append(list(temp_list))
            
            threads_list = []
            
        else:
            
            #process = multiprocessing.Process(target = process_model, 
            #                                 args = (procs,model))

            thread = threading.Thread(target = process_model, 
                                        args = (num_threads, temp_list))
            threads_list.append(thread)
    
if __name__ == "__main__":
    
    num_cpus = 5
    print(f"Processor count = {num_cpus}")
    
    processes_list = []

    matrix_manager = multiprocessing.Manager()
    #temp_matrix = matrix_manager.list()
    matrices_dict = matrix_manager.dict()

    for i in range(num_cpus):
        matrices_dict[i] = matrix_manager.list()

    for cpu_num in range(num_cpus):
        # for each CPU load the model (which it does automatically) and create threads
        # it wont go back to main so put the thread creation in a function
        process = multiprocessing.Process(target = create_threads, args=(num_cpus, matrices_dict[cpu_num]))
        processes_list.append(process)

    
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

    print(len(temp_matrix))
    print(temp_matrix)

    for i in range(len(temp_matrix)):

        print(len(temp_matrix[i]))
    
    #time.sleep(10)

    #ne_list = list(temp_matrix)
    #print(ne_list)

    #print(type(temp_matrix[0]))

# Check which process was started - and if all 3 are running, 
# how to avoid cannot start a process twice - test with dummy function