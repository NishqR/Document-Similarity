from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.

from time import time
start_nb = time()

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

start = time()
import os

from gensim.models import Word2Vec
if not os.path.exists('wmd/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
    

model = Word2Vec.load_word2vec_format('wmd/GoogleNews-vectors-negative300.bin.gz', binary=True)
#model = Word2Vec('wmd/GoogleNews-vectors-negative300.bin.gz')
print('Cell took %.2f seconds to run.' % (time() - start))

sentence_obama = 'Obama speaks to the media in Illinois'
sentence_president = 'The president greets the press in Chicago'
sentence_obama = sentence_obama.lower().split()
sentence_president = sentence_president.lower().split()

# Remove stopwords.
stop_words = stopwords.words('english')
sentence_obama = [w for w in sentence_obama if w not in stop_words]
sentence_president = [w for w in sentence_president if w not in stop_words]


distance = model.wmdistance(sentence_obama, sentence_president)
print(distance)