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

def calculate_jaccard(word_tokens1, word_tokens2):
    # Combine both tokens to find union.
    both_tokens = word_tokens1 + word_tokens2
    union = set(both_tokens)

    # Calculate intersection.
    intersection = set()
    for w in word_tokens1:
        if w in word_tokens2:
            intersection.add(w)

    jaccard_score = len(intersection)/len(union)
    return jaccard_score

def process_jaccard_similarity(base_document, documents):
    start = time()
    # Tokenize the base document we are comparing against.
    base_tokens = preprocess(base_document)
    documents = [documents]

    # Tokenize each document
    all_tokens = []
    for i, document in enumerate(documents):
        tokens = preprocess(document)
        all_tokens.append(tokens)

        #print("making word tokens at index:", i)

    all_scores = []
    for tokens in all_tokens:
        score = calculate_jaccard(base_tokens, tokens)

        all_scores.append(score)

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(all_scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    if highest_score > 0.95:
        highest_score = 0

    print('Cell took %.2f seconds to run.' % (time() - start))
    return highest_score

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
        #article_comp = preprocess(article_comp)

        for article_against in all_articles:

            #article_against = preprocess(article_against)

            score = process_jaccard_similarity(article_comp, article_against)
            count[0] += 1
            print(f"Count - {count[0]}/16129")
            temp_list.append(score)

        matrices_dict[cpu_num].append(list(temp_list))
        temp_list = []



def create_threads(articles_batch, all_articles, matrices_dict, cpu_num, count):

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

    if True:
        
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
        plt.savefig('base_matrix.png')
        #plt.show()

    
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
        end_index = int((i+1)*(len(articles_df)/num_cpus))
        #print(end_index)
        #print(num_list[start_index:end_index])

        process = multiprocessing.Process(target = create_threads, args=(articles[start_index:end_index], articles, matrices_dict, cpu_num, count))
        processes_list.append(process)
        start_index = int((i+1)*(len(articles_df)/num_cpus))

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

    wmd_matrix = temp_matrix
    
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
    plt.savefig('jaccard.png')

    wmd_diff_matrix = []
    for i in range(len(matrix)):
        
        temp_list = []
        for j in range(len(matrix[i])):
            temp_list.append(matrix[i][j] - wmd_matrix[i][j])
        
        wmd_diff_matrix.append(temp_list)
        
    plt.figure(figsize = (10,10))
    plt.set_cmap('autumn')

    plt.matshow(wmd_diff_matrix, fignum=1)

    plt.savefig('jaccard_diff.png')