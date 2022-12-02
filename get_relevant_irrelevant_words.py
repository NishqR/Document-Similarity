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

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('omw-1.4')
#nltk.download('punkt')

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

def process_model(words_batch, embeddings_dict, count):

    for word_ in words_batch:

        print(f"Doing embeddings for word {count[0]}")
        start = time()

        embeddings_dict[word_] = model.encode(word_)

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))



def create_threads(words_batch, embeddings_dict, count):

    
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
                                        args = (words_batch, embeddings_dict, count))
            threads_list.append(thread)

def multiprocess_embeddings(num_cpus, words_list):

    num_cpus = 6
    #num_cpus = multiprocessing.cpu_count()
    print(f"Processor count = {num_cpus}")
    
    num_threads = len(words_list) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    
    start_index = 0

    embeddings_manager = multiprocessing.Manager()
    
    embeddings_dict = embeddings_manager.dict()
    #count_dict = embeddings_manager.dict()

    count = embeddings_manager.list()
    count.append(0)
    
    for cpu_num in range(num_cpus):
        end_index = int((cpu_num+1)*(len(words_list)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_threads, args=(words_list[start_index:end_index], embeddings_dict, count))
        processes_list.append(process)
        start_index = int((cpu_num+1)*(len(words_list)/num_cpus))

    print(f"Processes list = {processes_list}")
    
    for process in processes_list:
        print(f"Starting process")
        process.start()

    for process in processes_list:
        print(f"Double checking process")
        process.join()

    return embeddings_dict

if __name__ == "__main__":

    run_relevant = False
    main_start = time() 

    #articles_df = pd.read_csv("all_articles.csv")
    articles_df = pd.read_csv("labelled_1.csv")
    articles_df.fillna("", inplace=True)

    num_runs = 0

    while(num_runs < 2):
        
        count_dict = {}


        if run_relevant == True:

            articles = list(articles_df[articles_df['relevant'] == 1]['text'])

        else:

            articles = list(articles_df[articles_df['relevant'] == 0]['text'])      

        #print(sorted(((v, k) for k, v in count_dict.items()), reverse=True))
        
        for article in articles:
        
        #start = time()
        
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
                    word_ = word_.replace("/", "")
                    word_ = word_.replace("-", "")
                    word_ = word_.replace("(", "")
                    word_ = word_.replace(")", "")

                    if lemmatizer.lemmatize(word_) != 'ha' and lemmatizer.lemmatize(word_) != 'wa':
                        word_ = lemmatizer.lemmatize(word_)

                    if word_ not in stop_words and word_ not in string.punctuation:
                        if word_ not in count_dict.keys():
                            #embeddings_dict[word_] = model.encode(word_)
                            #embeddings_dict[word_] = ""
                            count_dict[word_] = 1

                        else:
                            count_dict[word_] += 1

            #count[0]+=1
            #print('Cell took %.2f seconds to run.' % (time() - start))

        if run_relevant == True:
            relevant_df = pd.DataFrame(pd.Series(count_dict), columns = ['frequency'])
            relevant_df.reset_index(inplace=True)
            relevant_df = relevant_df.rename(columns={'index': 'word'})
            relevant_df = relevant_df.sort_values(by=['frequency'], ascending=False)
            
            # apply normalization techniques
            apply_scale_column = 'frequency'
            relevant_df['frequency_scaled'] = MinMaxScaler().fit_transform(np.array(relevant_df[apply_scale_column]).reshape(-1,1))
            
            #relevant_df['frequency_scaled'] = MinMaxScaler().fit_transform(np.array(irrelevant_df[apply_scale_column][0:len(relevant_df)]).reshape(-1,1))
              
            relevant_df.to_csv("relevant_words_count.csv")
            
        else:
            irrelevant_df = pd.DataFrame(pd.Series(count_dict), columns = ['frequency'])
            irrelevant_df.reset_index(inplace=True)
            irrelevant_df = irrelevant_df.rename(columns={'index': 'word'})
            irrelevant_df = irrelevant_df.sort_values(by=['frequency'], ascending=False)
            
            # apply normalization techniques
            apply_scale_column = 'frequency'
            irrelevant_df['frequency_scaled'] = MinMaxScaler().fit_transform(np.array(irrelevant_df[apply_scale_column]).reshape(-1,1))
            
            irrelevant_df.to_csv("irrelevant_words_count.csv")
        
        num_runs += 1
        run_relevant = True

    num_words_relevant = 70
    num_words_irrelevant = 70

    relevant_words = list(relevant_df.head(num_words_relevant).word)
    irrelevant_words = list(irrelevant_df.head(num_words_irrelevant).word)
    common_words = intersection(relevant_words, irrelevant_words)
    
    unique_relevant = [x for x in relevant_words if x not in common_words]
    unique_irrelevant = [x for x in irrelevant_words if x not in common_words]

    print("COMMON WORDS")
    print(common_words)
    
    print("UNIQUE RELEVANT")
    print(unique_relevant)

    print("UNIQUE IRRELEVANT")
    print(unique_irrelevant)

    relevant_embeddings_dict = multiprocess_embeddings(6, unique_relevant)
    
    relevant_df['relevancy_score'] = np.zeros(len(relevant_df))
    relevant_df['weighted_score'] = np.ones(len(relevant_df))

    for word_comp in unique_relevant:
        for word_against in unique_relevant:
            relevant_df.loc[relevant_df['word'] == word_comp, ['relevancy_score']] += (cosine_similarity([relevant_embeddings_dict[word_comp]], [relevant_embeddings_dict[word_against]]).flatten())

    relevant_df['weighted_score'] = relevant_df['relevancy_score'] * relevant_df['frequency_scaled']
    relevant_df = relevant_df.sort_values(by=['weighted_score'], ascending=False)
    relevant_df.head(num_words_relevant - len(common_words)).to_csv("relevant_test.csv")
    
    irrelevant_embeddings_dict = multiprocess_embeddings(6, unique_irrelevant)
    irrelevant_df['relevancy_score'] = np.zeros(len(irrelevant_df))
    irrelevant_df['weighted_score'] = np.ones(len(irrelevant_df))

    for word_comp in unique_irrelevant:
        for word_against in unique_irrelevant:
            irrelevant_df.loc[irrelevant_df['word'] == word_comp, ['relevancy_score']] += (cosine_similarity([irrelevant_embeddings_dict[word_comp]], [irrelevant_embeddings_dict[word_against]]).flatten())

    irrelevant_df['weighted_score'] = irrelevant_df['relevancy_score'] * irrelevant_df['frequency_scaled']
    irrelevant_df = irrelevant_df.sort_values(by=['weighted_score'], ascending=False)
    irrelevant_df.head(num_words_irrelevant - len(common_words)).to_csv("irrelevant_test.csv")

    #print(embeddings_dict)
    #print(len(list(embeddings_dict.keys())))
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))