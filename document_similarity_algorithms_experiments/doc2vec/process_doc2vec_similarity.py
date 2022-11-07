from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity

import string
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

base_document = """California American Water has filed an application with the State Water Resources Control Board to help provide financial support to customers who were unable to pay their water bills during the coronavirus pandemic. The funding will enable the utility to forgive past-due balances incurred by its customers between March 2020 and June 2021.

California American Water requested $6.8 million in relief for customers across its California service areas. The company anticipates receiving final approval by the State Water Resources Control Board around the first of the year. Customers do not have to apply for debt forgiveness. If approved, California American Water will apply any credits to affected customers' accounts within 60 days after receiving funding from the state, which is expected to arrive by February 2022.

"As we continue to see the effects from the ongoing pandemic, we understand that our customers may still be struggling financially," said Kevin Tilden, President of California American Water. "We hope this debt relief will ease some of the burden our customers face as a result of the COVID emergency."

This relief is another step California American Water has taken to provide financial aid to its customers. Customers with remaining balances pre-dated to the pandemic or have accrued after June 15, 2021, can utilize interest and penalty-free payment plans or extensions, preventing them from experiencing disruptions in service after the moratorium on disconnections for non-payment expires. Eligible customers can also enroll in California American Water's Assistance Program, which provides qualifying customers a discount on their monthly service charge.

About California American Water: California American Water, a subsidiary of American Water (NYSE: AWK), provides high-quality and reliable water and/or wastewater services to more than 880,000 California residents. Information regarding California American Water's service areas can be found on the company's website www.californiaamwater.com.

About American Water

With a history dating back to 1886, American Water (NYSE:AWK) is the largest and most geographically diverse U.S. publicly traded water and wastewater utility company. The company employs approximately 6,400 dedicated professionals who provide regulated and regulated-like drinking water and wastewater services to an estimated 14 million people in 25 states. American Water provides safe, clean, affordable and reliable water services to our customers to help keep their lives flowing. For more information, visit amwater.com and follow American Water on Twitter, Facebook and LinkedIn."""
documents = ["""Sen. John W. Hickenlooper (D-CO) News Release

U.S. Senate Documents

Denver, Colo.-Today, Senator John Hickenlooper applauded Connect for Health Colorado's announcement that 198,412 Coloradans signed up for a health insurance plan by the end of the most recent Open Enrollment Period. This marks an increase of more than 18,000 enrollments, or 10 percent, above last year's end of Open Enrollment total, and is the highest end of Open Enrollment total since Connect for Health Colorado opened in 2013.

Three out of four customers who signed up for a plan qualified for health insurance savings via premium subsidies passed by Congress as part of the American Rescue Plan. Thanks to these subsidies, individuals can save an average of 52 percent on their monthly premium.

While the annual enrollment period has ended, uninsured residents who have been impacted by the recent fires in Boulder County or by COVID-19 can access a new Disaster Relief Special Enrollment Period now through March 16, 2022.

"These numbers prove Connect for Health Colorado is a success story," said Colorado Senator John Hickenlooper. "Congress needs to keep premiums low to make health care an affordable reality for every Coloradan."

"The record level of enrollments we're seeing tells me that people are finding plans in their budget and that our sign-up process is smoother than ever before," said Chief Executive Officer Kevin Patterson.

"While the regular Open Enrollment Period is complete, we opened our doors again. We're here to make sure those who find themselves affected by the recent Marshall Fire and the pandemic can get the coverage and care they need," Patterson said.

"The American Rescue Plan helped make health care more affordable and accessible for nearly 200,000 Coloradans. That is what progress looks like," said Colorado Senator Michael Bennet. "As we continue to face ongoing challenges as a result of COVID-19 and the Marshall Fire, I'm grateful that the enrollment deadline has been extended to ensure more families can get covered this year."

Additional Enrollment Opportunities and Assistance

Qualifying events, like certain household changes or losing health insurance from a job, also open a 60-day Special Enrollment Period for residents who need health insurance or want to change their plan. People can enroll in Health First Colorado (Colorado's Medicaid program) and the Child Health Plan Plus (CHP+) program any time during the year if they qualify.

Free help is available year-round from enrollment experts--certified Brokers and community-based Assisters-- located throughout Colorado. Connect for Health Colorado offers Enrollment Centers to provide help signing up for a health insurance plan in person, virtually or by phone.

To apply for health insurance and to find local enrollment assistance contact Connect for Health Colorado at ConnectforHealthCO.com or by calling 855-752-6749 ."""]

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

def process_doc2vec_similarity():

    # Both pretrained models are publicly available at public repo of jhlau.
    # URL: https://github.com/jhlau/doc2vec

    # filename = './models/apnews_dbow/doc2vec.bin'
    filename = './models/enwiki_dbow/doc2vec.bin'

    model= Doc2Vec.load(filename)

    tokens = preprocess(base_document)

    # Only handle words that appear in the doc2vec pretrained vectors. enwiki_ebow model contains 669549 vocabulary size.
    tokens = list(filter(lambda x: x in model.wv.vocab.keys(), tokens))

    base_vector = model.infer_vector(tokens)

    vectors = []
    for i, document in enumerate(documents):

        tokens = preprocess(document)
        tokens = list(filter(lambda x: x in model.wv.vocab.keys(), tokens))
        vector = model.infer_vector(tokens)
        vectors.append(vector)

        print("making vector at index:", i)

    scores = cosine_similarity([base_vector], vectors).flatten()

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    most_similar_document = documents[highest_score_index]
    print("Most similar document by Doc2vec with the score:", most_similar_document, highest_score)

process_doc2vec_similarity()
