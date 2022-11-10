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

print("---------------------------- LOADING MODEL---------------------------- ")
start = time()
global model
#model = 10000
model = gensim.models.KeyedVectors.load_word2vec_format('wmd/GoogleNews-vectors-negative300.bin.gz', binary=True)
print('Cell took %.2f seconds to run.' % (time() - start))

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def process_wmd_similarity(article_comp, article_against, temp_scores_list):
    start = time()
    #base_document = preprocess(base_document)
    #documents = preprocess(documents[0])
    
    distance = model.wmdistance(article_comp, article_against)
    #distance = model
    print(f"Distance = {distance}")
    print('Cell took %.2f seconds to run.' % (time() - start))
    
    try: 
        score = 1 / distance
    except:
        score = 0

    temp_scores_list.append(score)
    print(f"Temp list = {temp_scores_list}")

def create_threads(articles_batch, all_articles, temp_scores_list):

    
    num_threads = 36
    threads_list = []

    for article_other in all_articles:
        article_against = preprocess(article_other)

        if len(threads_list) == num_threads:

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

            thread = threading.Thread(target = process_wmd_similarity,
                                        args = (article_comp, article_against, temp_scores_list))
            threads_list.append(thread)



if __name__ == "__main__":

    if True:
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        nltk.download('punkt')

        stop_words = stopwords.words('english')

        lemmatizer = WordNetLemmatizer()


        all_lines = []
        directory = 'factiva_articles/'
        for file in os.listdir(directory):
            
            f = open(directory + file, "r")
            lines = f.readlines()
            
            all_lines.extend(lines)
            

        # Extracting individual articles from the entire corpus
        total_article_count = 0
        articles_used = 0

        # Temporary string for storing each article
        temp_string = """"""

        # List of all articles
        all_articles = []

        # Temporary list that stores each article as separate lines in order to determine the article's line count
        temp_lines = []


        for line in all_lines:
            
            # "Document " followed by a code signals the end of an article
            if 'Document ' in line: 
                
                total_article_count += 1
                # If this article's line count is greater than 10, only then save it
                if len(temp_lines) > 10:
                    
                    # Keeping count of the number of articles used
                    articles_used += 1
                    
                    #print(len(temp_lines))
                    all_articles.append(temp_string)
                    
                # Reset temp variables
                temp_lines = []
                temp_string = """"""
            
            # Keep updating temp variables
            else:
                temp_string += line
                temp_lines.append(line)
                
        # Print count of articles        
        print(total_article_count)
        print(articles_used)


        # Sanity check and removing unnecessary information from articles
        sanitized_article_count = 0
        all_articles_sanitized = []

        for article in all_articles:
            
            # Get the first double line break index after the initial line breaks (starting after index 10)
            start_index = article.index("\n\n", 10)
            # Get the next double line break that's the start_index + 4 (4 characters - \n\n)
            end_index = article.index("\n\n", start_index + 4)
            
            # Sanity_check to make sure we're removing the correct information and not important information in the article
            if '504' in article[start_index:end_index]:
                sanitized_article_count += 1
            else: 
                print(start_index)
                print(end_index)
                print(article[start_index:end_index])
                #print(article)
                
            # Create new string that removes this substring from start_index to end_index
            new_article = article[0 : start_index : ] + article[end_index + 1 : :]
            
            # Replace all new lines with spaces
            new_article = new_article.replace('\n', ' ')
            
            new_article = new_article.strip(" ")
            
            # Add this article to list of sanitized articles
            all_articles_sanitized.append(new_article)
            
        print(sanitized_article_count)


        # Loading financial wellbeing df
        finance_df = pd.read_excel("Fin_Wellbeing_subsample.xlsx")
        finance_df = finance_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Matching article titles from financial wellbeing excelsheet to articles in sanitized articles list
        title_found = False
        article_found_count = 0

        article_dicts = [] 
        for article_title in finance_df['article_title']:
            
            article_title = article_title.strip(" ")
            article_title_list = article_title.split(" ")
            title_first_word = article_title_list[0]   
            title_found = False
            print("")
            print("-------------------------------NEW ARTICLE----------------------------------------")
            print("Article title: "+article_title)
            
            
            for article in all_articles_sanitized:
                
                article_list = article.split(" ")
                
                first_word_found_before = False
                for word in article_list[0:30]:
                    
                    if title_found == True:
                        break
                        
                    first_word_similarity = similar(title_first_word, word)
                    
                    if first_word_similarity > 0.9:
                        
                        print(f"FIRST WORD MATCH FOUND: {title_first_word}, {word}, {first_word_similarity}")
                        print(title_first_word, word, first_word_similarity)
                        
                        if first_word_found_before == False:
                            start_index = article.index(word)
                            
                        else:
                            start_index = article[start_index+len(word):].index(word)
                        
                        title_similarity = similar(article_title, article[start_index:start_index + len(article_title)])
                        print("Sentence:       "+ article[start_index:start_index + len(article_title)])
                        print("Sentence Similarity: " +str(title_similarity))
                        
                        first_word_found_before = True
                        if title_similarity > 0.9:
                            title_found = True
                            article_found_count += 1
                            new_dict = {}
                            
                            new_dict["article_title"] = article_title
                            new_dict["text"] = article
                            #new_dict["preprocessed_text"] = preprocess(artic)
                            #print(finance_df[finance_df["article_title"] == article_title]["relevant_campbell"])
                            new_dict["relevant_campbell"] = finance_df[finance_df["article_title"] == article_title]["relevant_campbell"].values[0]
                            new_dict["relevant_kristen"] = finance_df[finance_df["article_title"] == article_title]["relevant_kristen"].values[0]
                            new_dict["mismatch"] = finance_df[finance_df["article_title"] == article_title]["mismatch"].values[0]
                            
                            article_dicts.append(new_dict)
                            break
                            
                        else:
                            title_found = False
                            
                    else:
                        title_found = False
                    
            print("-----------------------------------------------------------------------------------")
            
        print(article_found_count)

        # Creating a new df with article titles, article text and classifications
        articles_df = pd.DataFrame(article_dicts)

        # sanity check to ensure all the matched values for classifications by campbell and kristen across both dataframes match
        for article_title in articles_df["article_title"]:
            if finance_df[finance_df["article_title"] == article_title]["relevant_campbell"].values[0] != articles_df[articles_df["article_title"] == article_title]["relevant_campbell"].values[0]:
                print("Error in relevant_campbell")
                print(article_title)
                print(finance_df[finance_df["article_title"] == article_title]["relevant_campbell"].values[0])
                print(articles_df[articles_df["article_title"] == article_title]["relevant_campbell"].values[0])

            if finance_df[finance_df["article_title"] == article_title]["relevant_kristen"].values[0] != articles_df[articles_df["article_title"] == article_title]["relevant_kristen"].values[0]:
                print("Error in relevant_kristen")
                print(article_title)
                print(finance_df[finance_df["article_title"] == article_title]["relevant_kristen"].values[0])
                print(articles_df[articles_df["article_title"] == article_title]["relevant_kristen"].values[0])

            if finance_df[finance_df["article_title"] == article_title]["mismatch"].values[0] != articles_df[articles_df["article_title"] == article_title]["mismatch"].values[0]:
                print("Error in mismatch")
                print(article_title)
                print(finance_df[finance_df["article_title"] == article_title]["mismatch"].values[0])
                print(articles_df[articles_df["article_title"] == article_title]["mismatch"].values[0])


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

        plt.show()

    manager = multiprocessing.Manager()
    
    articles = list(articles_df['text'])
    temp_matrix = []

    max_similarity = manager.list()
    max_similarity.append(0)
    min_similarity = manager.list()
    min_similarity.append(0)
    
    count = manager.list()
    count.append(0)
    
    num_cpus = 5
    #num_cpus = multiprocessing.cpu_count()
    print(f"Processor count = {num_cpus}")
    
    num_threads = len(articles_df) / num_cpus
    print(f"Thread count = {num_threads}")

    processes_list = []
    print(f"Processes list = {processes_list}")

    
    
    for index, row in articles_df.iterrows():

        print(f"Article Index = {index}")
        temp_scores_list = manager.list()

        print(f"Temp list = {temp_scores_list}")
        article_comp = preprocess(row['text'])



        for article_other in articles:
            
            print(f"Jobs list = {processes_list}")
            article_against = preprocess(article_other)
                    
            if len(processes_list) == num_cpus:

                print(f"Jobs list full")

                # Start the processes       
                for process in processes_list:
                    print(f"Starting jobs")
                    process.start()

                # Ensure all of the processes have finished
                for process in processes_list:
                    print(f"Double checking jobs")
                    process.join()
                
                processes_list = []
                
            else:
                
                process = multiprocessing.Process(target = process_wmd_similarity, 
                                                 args = (article_comp, article_against, temp_scores_list, max_similarity, min_similarity, count, model))
                processes_list.append(process)

        temp_matrix.append(temp_scores_list)

    print("Max similarity: " +str(max_similarity[0]))

    wmd_matrix = []

    for temp_scores_list in temp_matrix:

        normalized_list = []

        for similarity_value in temp_scores_list:

            normalized_similarity = similarity_value / max_similarity[0]
            normalized_list.append(normalized_similarity)

        wmd_matrix.append(normalized_list)

    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(wmd_matrix, fignum=1)

    plt.show()