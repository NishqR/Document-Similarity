import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

temp_matrix = np.load('results/tf_idf/tf_idf_matrix.npy')
articles_df = pd.read_csv("all_articles.csv")
print(temp_matrix)
plt.figure(figsize = (10,10))
plt.set_cmap('autumn')

plt.matshow(temp_matrix, fignum=1)
plt.savefig('results/tf_idf/temp_matrix.png')

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



temp_diff_matrix = []
for i in range(len(matrix)):
    
    temp_list = []
    for j in range(len(matrix[i])):
        temp_list.append(matrix[i][j] - temp_matrix[i][j])
    
    temp_diff_matrix.append(temp_list)
    
plt.figure(figsize = (10,10))
plt.set_cmap('autumn')

plt.matshow(temp_diff_matrix, fignum=1)

plt.savefig('results/tf_idf/temp_matrix_diff.png')


# EVALUATION SUITE
relevance_threshold = 0.6
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

total_relevant = 0
total_irrelevant = 0
for i in range(len(matrix)):
    
    for j in range(len(matrix[i])):

        if matrix[i][j] == 1:

            total_relevant += 1

            if temp_matrix[i][j] > relevance_threshold:
                true_positive += 1

            if temp_matrix[i][j] < relevance_threshold:
                false_negative += 1

        if matrix[i][j] == 0:

            total_irrelevant += 1

            if temp_matrix[i][j] > relevance_threshold:
                false_positive += 1

            if temp_matrix[i][j] < relevance_threshold:
                true_negative += 1

eval_dict = {}

print(f"Total relevant = {total_relevant}")
print(f"Total total_irrelevant = {total_irrelevant}")

print(f"True positive = {true_positive}")
print(f"True Negative = {true_negative}")
print(f"False_negative = {false_negative}")
print(f"false_positive = {false_positive}")

eval_dict['true_positive'] = true_positive
eval_dict['true_negative'] = true_negative
eval_dict['false_positive'] = false_positive
eval_dict['false_negative'] = false_negative
eval_dict['accuracy'] = (true_positive+true_negative)/(true_positive+true_negative+false_negative+false_positive)
eval_dict['precision'] = true_positive / (true_positive + false_positive)
eval_dict['recall'] = true_positive / (true_positive + false_negative)
eval_dict['f1_score'] = (2 * eval_dict['precision'] * eval_dict['recall']) / (eval_dict['precision'] + eval_dict['recall']) 
#print(f"correctly_classified documents  = {correctly_classified}")
print(f"Accuracy = {eval_dict['accuracy']}")
print(f"Precision = {eval_dict['precision']}")
print(f"Recall = {eval_dict['recall']}")
print(f"F1-Score = {eval_dict['f1_score']}")

eval_df = pd.DataFrame (pd.Series(eval_dict)).T
eval_df.to_csv('results/tf_idf/temp_metrics.csv')