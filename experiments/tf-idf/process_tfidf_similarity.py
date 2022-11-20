from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_document = """California American Water has filed an application with the State Water Resources Control Board to help provide financial support to customers who were unable to pay their water bills during the coronavirus pandemic. The funding will enable the utility to forgive past-due balances incurred by its customers between March 2020 and June 2021.

California American Water requested $6.8 million in relief for customers across its California service areas. The company anticipates receiving final approval by the State Water Resources Control Board around the first of the year. Customers do not have to apply for debt forgiveness. If approved, California American Water will apply any credits to affected customers' accounts within 60 days after receiving funding from the state, which is expected to arrive by February 2022.

"As we continue to see the effects from the ongoing pandemic, we understand that our customers may still be struggling financially," said Kevin Tilden, President of California American Water. "We hope this debt relief will ease some of the burden our customers face as a result of the COVID emergency."

This relief is another step California American Water has taken to provide financial aid to its customers. Customers with remaining balances pre-dated to the pandemic or have accrued after June 15, 2021, can utilize interest and penalty-free payment plans or extensions, preventing them from experiencing disruptions in service after the moratorium on disconnections for non-payment expires. Eligible customers can also enroll in California American Water's Assistance Program, which provides qualifying customers a discount on their monthly service charge.

About California American Water: California American Water, a subsidiary of American Water (NYSE: AWK), provides high-quality and reliable water and/or wastewater services to more than 880,000 California residents. Information regarding California American Water's service areas can be found on the company's website www.californiaamwater.com.

About American Water

With a history dating back to 1886, American Water (NYSE:AWK) is the largest and most geographically diverse U.S. publicly traded water and wastewater utility company. The company employs approximately 6,400 dedicated professionals who provide regulated and regulated-like drinking water and wastewater services to an estimated 14 million people in 25 states. American Water provides safe, clean, affordable and reliable water services to our customers to help keep their lives flowing. For more information, visit amwater.com and follow American Water on Twitter, Facebook and LinkedIn."""
documents = ["""SAINSBURY'S has launched a deal today where you can get 25% off the price of booze.

The discount applies if you buy six or more bottles of wine, champagne or prosecco and you can claim the money-off orders all week.

The deal on wine, champagne and prosecco began today and it is running until Sunday (May 2).

The store is offering the deal just in time the bank holiday weekend that falls at the beginning of May.

In the deal, a 375ml bottle of posh Moet costs £17.25 with the discount - saving you £5.75 on the full price of £23.

Of course, you'll need to buy six bottles for the deal to kick in. Normally, six bottles would set you back £138 but in the offer the booze costs £103.50.

Alternatively, if you're more of a prosecco fan you can get 750ml bottles of Freixenet Prosecco for £9 each, saving you £3 per bottle.

Normally a box of six would cost £72 but the offer brings the price down to £54.

And if you're more of a fan of wine, a 750ml bottle of Oyster Bay Sauvignon Blanc was £10, but now it's down to just £7.50 when you buy a crate of six.

The 25% discount excludes any bottles that are under £5 in England or £7 in Wales.

And it also doesn't apply for bottles that are 200ml and under, as well as all Sainsbury’s House wine, all fortified wine, boxed wine and gift sets.

It's worth bearing in mind that the deal is only worth it if you were already planning on buy six bottles of plonk, other wise you're spending more money than you need to.

On Sainsburys' website you can browse the wine selection where the qualifying products have a little red offer indicator displayed, which will help you determine what you can get at the bargain prices.

The deal is available online and in stores although online shoppers will need to factor in the delivery costs, which start at £1.

Or you can use the store's click and collect service thats free over £40 or £4 for anything cheaper, and the option of simply heading to your nearest supermarket is available as well.

Sainsburys has over 600 supermarkets in the UK and you can cash in on the deal at participating stores using the helpful locator tool to find out where your closest one is.

We've revealed six secrets to bagging ‘yellow sticker’ food from supermarkets including Sainsburys, and Tesco, Asda and Aldi too.

We also have the low down on Sainsburys' opening hours and everything you need to know about ordering online with the supermarket.

But the store has been in hot water lately as shoppers were left furious over the mystery of ‘missing’ Nectar bonus points."""]

def process_tfidf_similarity():
    vectorizer = TfidfVectorizer()

    # To make uniformed vectors, both documents need to be combined first.
    documents.insert(0, base_document)
    embeddings = vectorizer.fit_transform(documents)

    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i


    most_similar_document = documents[highest_score_index]

    print("Most similar document by TF-IDF with the score:", most_similar_document, highest_score)

process_tfidf_similarity()
