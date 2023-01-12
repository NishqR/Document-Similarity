import os
import string
from time import time
import random
import copy


import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from csv import DictWriter


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
relevant_stop_words = set(stopwords.words('english') + [word.strip("\n") for word in open("remove_words_from_relevant.txt", "r")])
irrelevant_stop_words = set(stopwords.words('english') + [word.strip("\n") for word in open("remove_words_from_irrelevant.txt", "r")])

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

def santize_word(word_):

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
    word_ = word_.replace("'", "")
    word_ = word_.replace("!", "")
    word_ = word_.replace("?", "")
    word_ = word_.replace(";", "")
    word_ = word_.replace("%", "")


    if lemmatizer.lemmatize(word_) != 'ha' and lemmatizer.lemmatize(word_) != 'wa':
        word_ = lemmatizer.lemmatize(word_)

    return word_

def process_use_model(words_batch, embeddings_dict, count):

    for word_ in words_batch:

        print(f"Doing embeddings for word {count[0]}")
        start = time()

        embeddings_dict[word_] = model([word_])

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))


def process_doc2vec_model(words_batch, embeddings_dict, count):

    for word_ in words_batch:

        print(f"Doing embeddings for word {count[0]}")
        start = time()

        tokens = list(filter(lambda x: x in model.wv.vocab.keys(), word_))
        base_vector = model.infer_vector(tokens)

        embeddings_dict[word_] = base_vector

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))



def process_bert_model(words_batch, embeddings_dict, count):

    for word_ in words_batch:

        print(f"Doing embeddings for word {count[0]}")
        start = time()

        embeddings_dict[word_] = model.encode(word_)

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))

def create_use_threads(words_batch, embeddings_dict, count):

    
    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    use_filename = "use/universal-sentence-encoder_4"
    model = hub.load(use_filename)
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
            thread = threading.Thread(target = process_use_model, 
                                        args = (words_batch, embeddings_dict, count))
            threads_list.append(thread)


def create_doc2vec_threads(words_batch, embeddings_dict, count):

    
    print("---------------------------- LOADING MODEL---------------------------- ")
    start = time()
    global model
    doc2vec_filename = 'doc2vec/doc2vec.bin'
    model= Doc2Vec.load(doc2vec_filename)
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
            thread = threading.Thread(target = process_doc2vec_model, 
                                        args = (words_batch, embeddings_dict, count))
            threads_list.append(thread)

def create_bert_threads(words_batch, embeddings_dict, count):

    
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
            thread = threading.Thread(target = process_bert_model, 
                                        args = (words_batch, embeddings_dict, count))
            threads_list.append(thread)

def multiprocess_embeddings(num_cpus, words_list):

    num_cpus = 6
    #num_cpus = multiprocessing.cpu_count()
    print(f"Processor count = {num_cpus}")
    
    num_threads = len(words_list) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    
    start_index_embedding = 0

    embeddings_manager = multiprocessing.Manager()
    
    embeddings_dict = embeddings_manager.dict()
    #count_dict = embeddings_manager.dict()

    count = embeddings_manager.list()
    count.append(0)
    
    for cpu_num in range(num_cpus):
        end_index_embedding = int((cpu_num+1)*(len(words_list)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_use_threads, args=(words_list[start_index_embedding:end_index_embedding], embeddings_dict, count))
        processes_list.append(process)
        start_index_embedding = int((cpu_num+1)*(len(words_list)/num_cpus))

    print(f"Processes list = {processes_list}")
    
    for process in processes_list:
        print(f"Starting process")
        process.start()

    for process in processes_list:
        print(f"Double checking process")
        process.join()

    return embeddings_dict

def process_classification(articles_batch, all_articles_relevancy, count, relevant_df, irrelevant_df):

    for index, row in articles_batch.iterrows():
            
        print(f"Classifying article {count[0]}")
        start = time()

        if len(list(row['text'].split(" "))) < 100:
            all_articles_relevancy[row['article_title']] = -10
            continue

        relevant_words_list = list(relevant_df['word'])
        irrelevant_words_list = list(irrelevant_df['word'])

        all_articles_relevancy[row['article_title']] = 0
        sentences = sent_tokenize(row['text'])

        for sentence in sentences:
            words_in_sentence = list(sentence.split(" "))
            for word_ in words_in_sentence:

                word_ = santize_word(word_)

                if (word_ not in stop_words) and (word_ not in string.punctuation):


                    if word_ in relevant_words_list:
                        #print("RELEVANT WORD")
                        #print(word_)
                        #print(relevant_df[relevant_df['word'] == word_]['weighted_score'])
                        all_articles_relevancy[row['article_title']] += float(relevant_df[relevant_df['word'] == word_]['weighted_score'])
                        #print(all_articles_relevancy['article_title'])

                    if word_ in irrelevant_words_list:
                        #print("IRRELEVANT WORD")
                        #print(word_)
                        #print(irrelevant_df[irrelevant_df['word'] == word_]['weighted_score'])
                        all_articles_relevancy[row['article_title']] -= float(irrelevant_df[irrelevant_df['word'] == word_]['weighted_score'])
                        #print(all_articles_relevancy['article_title'])

        count[0]+=1
        print('Cell took %.2f seconds to run.' % (time() - start))


def create_classification_threads(articles_batch, all_articles_relevancy, count, relevant_df, irrelevant_df):

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
            thread = threading.Thread(target = process_classification, 
                                        args = (articles_batch, all_articles_relevancy, count, relevant_df, irrelevant_df))
            threads_list.append(thread)

def multiprocess_classification(num_cpus, articles_df, relevant_df, irrelevant_df):

    print(f"Processor count = {num_cpus}")
    
    num_threads = len(articles_df) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    
    start_index_classification = 0

    articles_manager = multiprocessing.Manager()
    
    all_articles_relevancy = articles_manager.dict()

    #count_dict = embeddings_manager.dict()

    count = articles_manager.list()
    count.append(0)
    
    for cpu_num in range(num_cpus):
        end_index_classification = int((cpu_num+1)*(len(articles_df)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_classification_threads, args=(articles_df[start_index_classification:end_index_classification], all_articles_relevancy, count, relevant_df, irrelevant_df))
        processes_list.append(process)
        start_index_classification = int((cpu_num+1)*(len(articles_df)/num_cpus))

    print(f"Processes list = {processes_list}")
    
    for process in processes_list:
        print(f"Starting process")
        process.start()

    for process in processes_list:
        print(f"Double checking process")
        process.join()

    return all_articles_relevancy

if __name__ == "__main__":

    main_start = time()

    folder_name = "fwb_data/"
    filename = "part-000000000000"
    filetype = ".json"
    articles_df = pd.read_json(folder_name+filename+filetype, lines=True)
    articles_df.fillna("", inplace=True)
    articles_df = articles_df.reset_index().drop(columns=['subject_codes','copyright','index','language_code', 'byline','company_codes_association', 
    'art', 'modification_datetime', 'company_codes_occur_ticker_exchange', 'company_codes_occur', 'company_codes_about', 'company_codes_lineage',
    'company_codes_ticker_exchange','company_codes_relevance_ticker_exchange','publication_date',    'market_index_codes','credit','section',
    'company_codes_association_ticker_exchange','currency_codes','an','word_count','company_codes','industry_codes','action','document_type','dateline', 
    'publication_datetime','company_codes_relevance','source_code','person_codes','company_codes_lineage_ticker_exchange','ingestion_datetime',
    'modification_date','company_codes_about_ticker_exchange'])
    articles_df['text'] = articles_df['title'] + ' ' + articles_df['snippet'] + ' ' + articles_df['body']
    articles_df = articles_df.rename(columns={'title': 'article_title', 'relevance': 'relevant'})
    articles_df = articles_df.reset_index().drop(columns=['snippet', 'body', 'index'])
    
    articles_df['text'] = articles_df.text.apply(lambda x: re.sub(r"\n", " ", str(x)))
    articles_df['text'] = articles_df.text.apply(lambda x: re.sub(r"\t", " ", str(x)))
    articles_df['article_title'] = articles_df.article_title.apply(lambda x: re.sub(r"\n", " ", str(x)))
    articles_df['article_title'] = articles_df.article_title.apply(lambda x: re.sub(r"\t", " ", str(x)))
    
    #articles_df['text'] = articles_df.text.apply(lambda x: re.sub(r",", " ", str(x)))

    #test_set=articles_df.sample(frac=0.1,random_state=195)
    relevant_df = pd.read_csv('relevant_words.csv')
    irrelevant_df = pd.read_csv('irrelevant_words.csv')

    # Input parameters
    num_classifiers = 1
    classifier_ = 0

    start_index = 0
    labelled_articles_df = pd.read_csv("all_articles.csv")
    labelled_articles_df.fillna("", inplace=True)
    #chunk_percentages = [1,2,4,4,4,4,4,4,4,4,20,20,25]
    chunk_percentages = [10]

    for chunk_percentage in chunk_percentages:
        

        #------------------------------------------------------------------------------------ CLASSIFICATION ------------------------------------------------------------------------------------#
        print(f"Classifying batch {chunk_percentage}")
        end_index = start_index + int((chunk_percentage/100) * len(articles_df))
        
        df_chunk = articles_df[start_index:end_index].copy(deep=True)
        
        df_chunk['relevant'] = np.zeros(len(df_chunk))
        df_chunk['relevancy_score'] = np.zeros(len(df_chunk))
        
        #for index, row in df[start_index:end_index].iterrows():
            #print(index)

        # Do classification based on number of relevant and irrelevant words
        all_articles_relevancy = multiprocess_classification(8, df_chunk, relevant_df, irrelevant_df)

        for key in all_articles_relevancy.keys():
            df_chunk.loc[df_chunk['article_title'] == key, ['relevancy_score']] = all_articles_relevancy[key]

            if all_articles_relevancy[key] >= 0:
                df_chunk.loc[df_chunk['article_title'] == key, ['relevant']] = 1

            
        # Concatenation
        concatenation = [labelled_articles_df, df_chunk]
        labelled_articles_df = pd.concat(concatenation,ignore_index=True)
        
        #print("----------------------------------------------------")
        start_index = end_index
        

    labelled_articles_df.to_csv("results/"+filename+".csv")
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))