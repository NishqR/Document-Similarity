import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from time import time

import gensim
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer

print("---------------------------- LOADING MODEL---------------------------- ")
start = time()
global model
filename = 'doc2vec/doc2vec.bin'
model= Doc2Vec.load(filename)
print('MODEL LOADED in %.2f seconds' % (time() - start))


similar_words =model.most_similar(positive=['pension'], topn = 3)

print(similar_words)
#from transformers import BertTokenizer, BertModel
#import torch
'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

word = "Hello"
inputs = tokenizer(word, return_tensors="pt")
outputs = model(**inputs)
word_vect = outputs.pooler_output.detach().numpy()

print(outputs)
print(word_vect)

'''
'''
model = Word2Vec(sentences=common_texts, window=5, min_count=1, workers=4)
base_vector = model.wv['computer']
comp_vector = model.wv['program']
print(cosine_similarity([base_vector], [comp_vector]).flatten())
'''
