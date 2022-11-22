import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


model_name = 'bert'
temp_matrix = np.load('results/'+model_name+'/'+model_name+'_matrix.npy')
articles_df = pd.read_csv("all_articles.csv")

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
plt.savefig('results/base_matrix.png')

max_similarity = 0
min_similarity = 10000

for list_ in temp_matrix:
    for val_ in list_:

        if val_ > max_similarity:
            max_similarity = val_

        if val_ < min_similarity:
            min_similarity = val_

model_matrix = []

for temp_scores_list in temp_matrix:

    normalized_list = []

    for similarity_value in temp_scores_list:

        normalized_similarity = similarity_value / max_similarity
        normalized_list.append(normalized_similarity)

    model_matrix.append(normalized_list)


np.save('results/'+model_name+'/'+model_name+'_matrix.npy', np.array(model_matrix))



plt.figure(figsize = (10,10))
plt.set_cmap('autumn')

plt.matshow(model_matrix, fignum=1)
plt.savefig('results/'+model_name+'/'+model_name+'.png')

model_diff_matrix = []
for i in range(len(matrix)):
    
    temp_list = []
    for j in range(len(matrix[i])):
        temp_list.append(matrix[i][j] - model_matrix[i][j])
    
    model_diff_matrix.append(temp_list)
    
plt.figure(figsize = (10,10))
plt.set_cmap('autumn')

plt.matshow(model_diff_matrix, fignum=1)

plt.savefig('results/'+model_name+'/'+model_name+'_diff.png')


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

            if model_matrix[i][j] > relevance_threshold:
                true_positive += 1

            if model_matrix[i][j] < relevance_threshold:
                false_negative += 1

        if matrix[i][j] == 0:

            total_irrelevant += 1

            if model_matrix[i][j] > relevance_threshold:
                false_positive += 1

            if model_matrix[i][j] < relevance_threshold:
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
eval_df.to_csv('results/'+model_name+'/'+model_name+'_metrics.csv')