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
from csv import DictWriter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize

#stop_words = stopwords.words('english')
stop_words = set(stopwords.words('english') + [word.strip("\n") for word in open("remove_words.txt", "r")])

lemmatizer = WordNetLemmatizer()

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


if __name__ == "__main__":

    main_start = time()

    test_set = pd.read_csv("")
    test_set.fillna("", inplace=True)

    relevant_df = pd.read_csv("relevant_test.csv")
    irrelevant_df = pd.read_csv("irrelevant_test.csv")

    num_classifiers = 1
    classifier_ = 0

    num_words_relevant = 60 # vary?
    num_words_irrelevant = 50 # vary?

    num_similar_relevant = 3
    similarity_threshold_relevant = 0.6

    num_similar_irrelevant = 1
    similarity_threshold_irrelevant = 0.8


    #------------------------------------------------------------------------------------ CLASSIFICATION ------------------------------------------------------------------------------------#

    test_set['predicted' + str(classifier_)] = np.zeros(len(test_set))
    all_articles_relevancy = {}
    for index, row in test_set.iterrows():
        
        #if index == 0:
        all_articles_relevancy[row['article_title']] = 0
        sentences = sent_tokenize(row['text'])

        for sentence in sentences:
            words_in_sentence = list(sentence.split(" "))
            for word_ in words_in_sentence:

                word_ = santize_word(word_)

                if (word_ not in stop_words) and (word_ not in string.punctuation):

                    if word_ in list(relevant_df['word']):
                        #print("RELEVANT WORD")
                        #print(word_)
                        #print(relevant_df[relevant_df['word'] == word_]['weighted_score'])
                        all_articles_relevancy[row['article_title']] += float(relevant_df[relevant_df['word'] == word_]['weighted_score'])
                        #print(all_articles_relevancy['article_title'])

                    if word_ in list(irrelevant_df['word']):
                        #print("IRRELEVANT WORD")
                        #print(word_)
                        #print(irrelevant_df[irrelevant_df['word'] == word_]['weighted_score'])
                        all_articles_relevancy[row['article_title']] -= float(irrelevant_df[irrelevant_df['word'] == word_]['weighted_score'])
                        #print(all_articles_relevancy['article_title'])

    for key in all_articles_relevancy.keys():
        if all_articles_relevancy[key] >= 0:
            test_set.loc[test_set['article_title'] == key, ['predicted' + str(classifier_)]] = 1


    test_set.to_csv("results/predicted_vals.csv")

    # Take mean across all predictions and save to predicted column
    all_predictions = []
    for index, row in test_set.iterrows():
        
        total_val = 0
        for classifier_ in range(num_classifiers):
            #print(classifier_)
            #print(row['predicted'+str(classifier_)])
            total_val += float(row['predicted'+str(classifier_)])
        
        avg_prediction = total_val/num_classifiers
        print(avg_prediction)
        
        if avg_prediction > 0.5:
            all_predictions.append(1)
        else:
            all_predictions.append(0)
            
    test_set['predicted'] = all_predictions

    #pd.set_option('display.max_rows', 1000)
    #test_set[['relevant', 'predicted']]

    
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
    with open('results/label_2_classification.csv', 'a') as f_object:
     
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = DictWriter(f_object, fieldnames=field_names)
     
        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(eval_dict)
     
        # Close the file object
        f_object.close()

    #eval_df.to_csv('results/classification.csv')
    
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))