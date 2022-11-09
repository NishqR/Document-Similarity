import os
import string
from time import time

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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize

import threading
import random

stop_words = stopwords.words('english')
    
print("---------------------------- LOADING MODEL---------------------------- ")
start = time()
global model
#model = 1000000
model = gensim.models.KeyedVectors.load_word2vec_format('wmd/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Cell took %.2f seconds to run.' % (time() - start))


def preprocess(sentence):

    sentence = sentence.lower().split()
    return [w for w in sentence.lower().split() if w not in stop_words]


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

def process_model(dummy, x):

    out_list = []
    for i in range(model):
        out_list.append(random.random())
'''

def create_threads():

    x = 5
    num_threads = 32
    threads_list = []

    print(f"Thread count = {num_threads}")
    for r in range(1000):
        #print(f"Jobs list = {jobs_list}")

        if len(threads_list) == num_threads:

            print(f"Threads list full")

            # Start the processes       
            for thread in threads_list:
                #print(f"Starting threads")
                thread.start()

            # Ensure all of the processes have finished
            for thread in threads_list:
                #print(f"Double checking threads")
                thread.join()
            
            threads_list = []
            
        else:
            
            #process = multiprocessing.Process(target = process_model, 
            #                                 args = (procs,model))

            thread = threading.Thread(target = process_model, 
                                        args = (num_threads, x))
            threads_list.append(thread)

if __name__ == "__main__":
        
    
    cpus = 3
    
    #procs = multiprocessing.cpu_count()
    
    print(f"Processor count = {cpus}")
    
    processes_list = []
    print(f"Processes list = {processes_list}")

    for i in range(cpus):
        # for each CPU load the model (which it does automatically) and create threads
        # it wont go back to main so put the thread creation in a function
        process = multiprocessing.Process(target = create_threads)
        processes_list.append(process)

    
    print(f"Processes list = {processes_list}")
    for process in processes_list:

        for process in processes_list:
            print(f"Starting process")
            process.start()

        for process in processes_list:
            print(f"Double checking process")
            process.join()


# Check which process was started - and if all 3 are running, 
# how to avoid cannot start a process twice - test with dummy function