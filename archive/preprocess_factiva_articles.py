import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from time import time
import random
from difflib import SequenceMatcher
from collections import defaultdict

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

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




articles_df.to_csv("all_articles.csv")
