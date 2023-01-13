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
    filename = "use/universal-sentence-encoder_4"
    model = hub.load(filename)
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
    filename = 'doc2vec/doc2vec.bin'
    model= Doc2Vec.load(filename)
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

        process = multiprocessing.Process(target = create_use_threads, args=(words_list[start_index:end_index], embeddings_dict, count))
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
    
    start_index = 0

    articles_manager = multiprocessing.Manager()
    
    all_articles_relevancy = articles_manager.dict()

    #count_dict = embeddings_manager.dict()

    count = articles_manager.list()
    count.append(0)
    
    for cpu_num in range(num_cpus):
        end_index = int((cpu_num+1)*(len(articles_df)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_classification_threads, args=(articles_df[start_index:end_index], all_articles_relevancy, count, relevant_df, irrelevant_df))
        processes_list.append(process)
        start_index = int((cpu_num+1)*(len(articles_df)/num_cpus))

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

    # Load all articles into dataframe
    articles_df = pd.read_csv("all_articles_updated.csv")
    articles_df.fillna("", inplace=True)

    # Split articles into train and test sets
    train_set=articles_df.sample(frac=1,random_state=200)
    test_set=articles_df.sample(frac=1,random_state=195)
    #test_set=articles_df.drop(train_set.index)

    # Input parameters
    num_classifiers = 1
    classifier_ = 0

    num_words_relevant = 110
    num_words_irrelevant = 110

    num_similar_relevant = 4
    similarity_threshold_relevant = 0.8

    num_similar_irrelevant = 3
    similarity_threshold_irrelevant = 0.8

    num_runs = 0

    run_relevant = True 

    if True:

        # Get the frequency of each word in the article post-sanitization and store it in a dataframe
        while(num_runs < 2):
            
            # Dictionary that stores the frequency of each word
            count_dict = {}

            # If we are getting the relevant words, take all the relevant and make a list of stop words that includes
            # common stop words + relevant-specific stop words
            if run_relevant == True:
                articles = list(train_set[train_set['relevant'] == 1]['text'])
                stop_words_specific = set(list(copy.deepcopy(stop_words)) + list(copy.deepcopy(relevant_stop_words)))

            # Else take all the irrelevant articles and make a list of stop words that includes
            # common stop words + irrelevant-specific stop words
            else:
                articles = list(train_set[train_set['relevant'] == 0]['text'])
                stop_words_specific = set(list(copy.deepcopy(stop_words)) + list(copy.deepcopy(irrelevant_stop_words)))      

            # Sanitize each word in each article, and save it to the count dictionary with frequency updating
            for article in articles:
            
                sentences = sent_tokenize(article)
                
                for sentence in sentences:
                    words_in_sentence = list(sentence.split(" "))
                    for word_ in words_in_sentence:

                        word_ = santize_word(word_)

                        if (word_ not in stop_words_specific) and (word_ not in string.punctuation):

                            if word_ not in count_dict.keys():
                                count_dict[word_] = 1

                            else:
                                count_dict[word_] += 1


            # Save the words to a dataframe
            if run_relevant == True:
                relevant_df_all = pd.DataFrame(pd.Series(count_dict), columns = ['frequency'])
                relevant_df_all.reset_index(inplace=True)
                relevant_df_all = relevant_df_all.rename(columns={'index': 'word'})
                relevant_df_all = relevant_df_all.sort_values(by=['frequency'], ascending=False)
                  
                
            else:
                irrelevant_df_all = pd.DataFrame(pd.Series(count_dict), columns = ['frequency'])
                irrelevant_df_all.reset_index(inplace=True)
                irrelevant_df_all = irrelevant_df_all.rename(columns={'index': 'word'})
                irrelevant_df_all = irrelevant_df_all.sort_values(by=['frequency'], ascending=False)
               
            
            num_runs += 1
            run_relevant = False

        # Scaling of the frequencies
        
        # First calculate the max frequency of each of the lists
        relevant_max_val = relevant_df_all['frequency'].max()
        irrelevant_max_val = irrelevant_df_all['frequency'].max()
        
        # Scale frequencies for relevant df
        normalized_freq = []
        for freq_ in list(relevant_df_all['frequency']):

            scaled_freq = freq_ / relevant_max_val
            normalized_freq.append(scaled_freq)

        relevant_df_all['frequency_scaled'] = normalized_freq

        # Scale frequencies for irrelevant df
        normalized_freq = []
        for freq_ in list(irrelevant_df_all['frequency']):

            scaled_freq = freq_ / irrelevant_max_val
            normalized_freq.append(scaled_freq)

        irrelevant_df_all['frequency_scaled'] = normalized_freq
        

        # Sort dataframes again by scaled frequencies 
        relevant_df = relevant_df_all.sort_values(by=['frequency_scaled'], ascending=False)
        irrelevant_df = irrelevant_df_all.sort_values(by=['frequency_scaled'], ascending=False)

        # Make a copy dataframe of the all the relevant and irrelevant words
        relevant_df = relevant_df_all.copy(deep=True)
        irrelevant_df = irrelevant_df_all.copy(deep=True)

        # Trim down the relevant and irrelevant words
        relevant_words = list(relevant_df.head(num_words_relevant).word)
        irrelevant_words = list(irrelevant_df.head(num_words_irrelevant).word)



        # Load the embedding model for generating similar words
        print("---------------------------- LOADING MODEL---------------------------- ")
        start = time()
        global model
        filename = 'doc2vec/doc2vec.bin'
        model= Doc2Vec.load(filename)
        print('MODEL LOADED in %.2f seconds' % (time() - start))

        # Generate embeddings for relevant words
        relevant_embeddings_dict = multiprocess_embeddings(6, relevant_words)
        relevant_df['relevancy_score'] = np.zeros(len(relevant_df))
        relevant_df['weighted_score'] = np.ones(len(relevant_df))

        # Calculate weighted score based on relevancy and scaled frequency from embeddings
        for word_comp in relevant_words:
            for word_against in relevant_words:
                relevant_df.loc[relevant_df['word'] == word_comp, ['relevancy_score']] += (cosine_similarity(relevant_embeddings_dict[word_comp], relevant_embeddings_dict[word_against]).flatten())

        relevant_df['weighted_score'] = relevant_df['relevancy_score'] * relevant_df['frequency_scaled']
        relevant_df = relevant_df.head(num_words_relevant)

        
        # Generate similar relevant words
        words_to_add = []

        for index, row in relevant_df.head(num_words_relevant).iterrows():
            
            try:
                similar_words = model.most_similar(positive=[row['word']], topn = num_similar_relevant)
            except:
                similar_words = []
            
            for similar_word in similar_words:
                word_ = similar_word[0]

                word_ = santize_word(word_)

                if (word_ not in stop_words) and (word_ not in string.punctuation) and (word_ not in relevant_stop_words):
                    if (word_ != (row['word'] + 's')) and (similar_word[1] >= similarity_threshold_relevant):
                        if (word_ not in words_to_add) and (word_ not in list(relevant_df['word'])):
                            words_to_add.append(word_)
                            temp_dict = {"word": word_, "frequency": row['frequency'], "frequency_scaled": row['frequency_scaled'], "relevancy_score": row['relevancy_score'], "weighted_score": float(similar_word[1]) * float(row["weighted_score"])}
                            relevant_df = relevant_df.append(temp_dict, ignore_index = True)
            
        relevant_df = relevant_df.sort_values(by=['weighted_score'], ascending=False)
        relevant_df.to_csv("relevant_words.csv")
        

        # Generate embeddings for irrelevant words
        irrelevant_embeddings_dict = multiprocess_embeddings(6, irrelevant_words)
        irrelevant_df['relevancy_score'] = np.zeros(len(irrelevant_df))
        irrelevant_df['weighted_score'] = np.ones(len(irrelevant_df))

        # Calculate weighted score based on relevancy and scaled frequency from embeddings
        for word_comp in irrelevant_words:
            for word_against in irrelevant_words:
                irrelevant_df.loc[irrelevant_df['word'] == word_comp, ['relevancy_score']] += (cosine_similarity(irrelevant_embeddings_dict[word_comp], irrelevant_embeddings_dict[word_against]).flatten())

        irrelevant_df['weighted_score'] = irrelevant_df['relevancy_score'] * irrelevant_df['frequency_scaled']
        irrelevant_df = irrelevant_df.head(num_words_irrelevant)

        # Generate similar irrelevant words
        words_to_add = []

        for index, row in irrelevant_df.head(num_words_irrelevant).iterrows():
            
            try:
                similar_words = model.most_similar(positive=[row['word']], topn = num_similar_irrelevant)
            except:
                similar_words = []
            
            for similar_word in similar_words:
                word_ = similar_word[0]

                word_ = santize_word(word_)

                if (word_ not in stop_words) and (word_ not in string.punctuation) and (word_ not in irrelevant_stop_words):
                    if (word_ != (row['word'] + 's')) and (similar_word[1] >= similarity_threshold_irrelevant):
                        if (word_ not in words_to_add) and (word_ not in list(irrelevant_df['word'])):
                            words_to_add.append(word_)
                            temp_dict = {"word": word_, "frequency": row['frequency'], "frequency_scaled": row['frequency_scaled'], "relevancy_score": row['relevancy_score'], "weighted_score": float(similar_word[1]) * float(row["weighted_score"])}
                            irrelevant_df = irrelevant_df.append(temp_dict, ignore_index = True)


        irrelevant_df = irrelevant_df.sort_values(by=['weighted_score'], ascending=False)
        irrelevant_df.to_csv("irrelevant_words.csv")


        #------------------------------------------------------------------------------------ CLASSIFICATION ------------------------------------------------------------------------------------#

        # Do classification based on number of relevant and irrelevant words
        test_set['predicted' + str(classifier_)] = np.zeros(len(test_set))
        test_set['relevancy_score'  + str(classifier_)] = np.zeros(len(test_set))
        all_articles_relevancy = multiprocess_classification(6, test_set, relevant_df, irrelevant_df)
        
        # Classify articles as relevant if their relevancy scores are greater than or equal to 0
        for key in all_articles_relevancy.keys():
            test_set.loc[test_set['article_title'] == key, ['relevancy_score']] = all_articles_relevancy[key]

            if all_articles_relevancy[key] >= 0:
                test_set.loc[test_set['article_title'] == key, ['predicted' + str(classifier_)]] = 1

        # Save the results to file
        test_set.to_csv("results/predicted_vals.csv")

        # Evaluation of the classifier

        # Take mean across all predictions and save to predicted column
        all_predictions = []
        for index, row in test_set.iterrows():
            
            total_val = 0
            for classifier_ in range(num_classifiers):
                total_val += float(row['predicted'+str(classifier_)])
            
            avg_prediction = total_val/num_classifiers
            print(avg_prediction)
            
            if avg_prediction > 0.5:
                all_predictions.append(1)
            else:
                all_predictions.append(0)
                
        test_set['predicted'] = all_predictions

        num_ers = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for index, row in test_set.iterrows():
            
            if row['relevant'] == 1:
                
                if row['predicted'] == 1:
                    true_positive += 1
                
                if row['predicted'] == 0:
                    false_negative += 1
            
            if row['relevant'] == 0:
                
                if row['predicted'] == 0:
                    true_negative += 1
                    
                if row['predicted'] == 1:
                    false_positive += 1
                
            
            if row['relevant'] != row['predicted']:
                num_ers += 1

        eval_dict = {}

        print(f"True positive = {true_positive}")
        print(f"True Negative = {true_negative}")
        print(f"False_negative = {false_negative}")
        print(f"False_positive = {false_positive}")
        print(f"Total number of errors = {num_ers}/{len(test_set)}")
        eval_dict['frac'] = "single"
        eval_dict['num_similar_relevant'] = num_similar_relevant
        eval_dict['num_similar_irrelevant'] = num_similar_irrelevant
        eval_dict['similarity_threshold_relevant'] = similarity_threshold_relevant
        eval_dict['similarity_threshold_irrelevant'] = similarity_threshold_irrelevant
        eval_dict['num_words_relevant'] = num_words_relevant
        eval_dict['num_words_irrelevant'] = num_words_irrelevant            
        eval_dict['true_positive'] = true_positive
        eval_dict['true_negative'] = true_negative
        eval_dict['false_positive'] = false_positive
        eval_dict['false_negative'] = false_negative
        
        try: 
            eval_dict['accuracy'] = (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
        except: 
            eval_dict['accuracy'] = 0

        try:
            eval_dict['precision'] = true_positive / (true_positive + false_positive)
        except:
            eval_dict['precision'] = 0
        
        try:
            eval_dict['recall'] = true_positive / (true_positive + false_negative)
        except: 
            eval_dict['recall'] = 0


        try:
            eval_dict['f1_score'] = (2 * eval_dict['precision'] * eval_dict['recall']) / (eval_dict['precision'] + eval_dict['recall']) 
        except:
            eval_dict['f1_score'] = 0

        #print(f"correctly_classified documents  = {correctly_classified}")
        print(f"Accuracy = {eval_dict['accuracy']}")
        print(f"Precision = {eval_dict['precision']}")
        print(f"Recall = {eval_dict['recall']}")
        print(f"F1-Score = {eval_dict['f1_score']}")

        eval_df = pd.DataFrame (pd.Series(eval_dict)).T

        # list of column names
        field_names = list(eval_dict.keys())
        
        # Open CSV file in append mode
        # Create a file object for this file
        with open('results/separate_freq.csv', 'a') as f_object:
         
            # Pass the file object and a list
            # of column names to DictWriter()
            # You will get a object of DictWriter
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
         
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(eval_dict)
         
            # Close the file object
            f_object.close()
            
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))