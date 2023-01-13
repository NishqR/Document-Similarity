import os
import string
from time import time
import random
import copy

stop_words = set(stopwords.words('english') + [word.strip("\n") for word in open("remove_words.txt", "r")])



def process_use_model(words_batch, embeddings_dict, count):

    for word_ in words_batch:

        print(f"Doing embeddings for word {count[0]}")
        start = time()

        embeddings_dict[word_] = model([word_])

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
    articles_df = pd.read_csv("all_articles.csv")
    articles_df.fillna("", inplace=True)

    # Split articles into train and test sets
    train_set=articles_df.sample(frac=0.8,random_state=200)
    test_set=articles_df.drop(train_set.index)

    # Trim down the relevant and irrelevant words
    relevant_words = list(relevant_df.head(num_words_relevant).word)
    irrelevant_words = list(irrelevant_df.head(num_words_irrelevant).word)

    # Generate embeddings for relevant words
    relevant_embeddings_dict = multiprocess_embeddings(6, relevant_words)    

    # Generate embeddings for irrelevant words
    irrelevant_embeddings_dict = multiprocess_embeddings(6, irrelevant_words)

    # Do classification based on number of relevant and irrelevant words
    all_articles_relevancy = multiprocess_classification(6, test_set, relevant_df, irrelevant_df)
        
    print('Script took %.2f minutes to run.' % ((time() - main_start)/60))