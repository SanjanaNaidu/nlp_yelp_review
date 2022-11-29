import json
import sys

import pandas

from Yelp_NLP.helper_functions import read_file_into_df, count_ngrams, print_word_cloud, print_most_frequent, \
    return_most_important_words, return_most_frequent_phrases
# import mysql.connector
# from mysql.connector import Error
import pymysql
from sqlalchemy import create_engine

# Reading the businesses file
# yelp_df = read_file_into_df(r"C:\Users\carls\PycharmProjects\ISProject\files\yelp_academic_dataset_review.json")
# yelp_df = yelp_df[["business_id", "stars", "text", "date"]]
# yelp_df.rename(columns={"text": "review_text", "date": "review_date"}, inplace=True)

# yelp_df['review length'] = yelp_df['text'].apply(len)
# sns.set_style('white')
dic = ["hi there", "hello asdf"]

dicc = {"asdf": 1, "vvvv": 2}
try:
    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                           .format(user="root",
                                   pw="Cookers00800*",
                                   db="nlp_yelp_reviews"))
    yelp_df = pandas.read_sql(sql="select business_id, stars, review_date, review_text from review_data", con=engine)
except Exception as e:
    print("Error while saving the data to MySQL", e)
    sys.exit()

business_ids_list = ["GBTPC53ZrG1ZBY3DT8Mbcw","PY9GRfzr4nTZeINf346QOw","SZU9c8V2GuREDN5KgyHFJw","UCMSWPqzXjd7QHq7v8PJjQ","vN6v8m4DO45Z4pp8yxxF_w","W4ZEKkva9HpAdZG88juwyQ","8uF-bhJFgT4Tn6DTb27viA","pSmOH4a3HNNpYM82J5ycLA","Zi-F-YvyVOK0k5QD7lrLOg","g04aAvgol7IW8buqSbT4xA","DcBLYSvOuWcNReolRVr12A","Zx7n8mdt8OzLRXVzolXNhQ","ORL4JE6tz3rJxVqkdKfegA","M0r9lUn2gLFYgIwIfG8-bQ","S2Ho8yLxhKAa26pBAm6rxA","EtKSTHV5Qx_Q7Aur9o4kQQ","2KIDQyTh-HzLxOUEDqtDBg","j-qtdD55OLfSqfsWuQTDJg","TV81bpCQ6p6o4Hau5hk-zw","mhrW9O0O5hXGXGnEYBVoag","cXSyVvOr9YRN9diDkaWs0Q","nRKndeZLQ3eDL10UMwS2rQ","EQ-TZ2eeD_E0BHuvoaeG5Q"]
for business_id in business_ids_list:

    # Filtering based on business ID
    yelp_bad_reviews = yelp_df.loc[(yelp_df.stars <= 2) & (yelp_df.business_id == business_id)]
    yelp_good_review = yelp_df.loc[(yelp_df.stars >= 4) & (yelp_df.business_id == business_id)]

    bad_reviews = yelp_bad_reviews.review_text
    good_reviews = yelp_good_review.review_text

    # Printing the word cloud and most frequent phrases in the bad reviews
    most_frequent_bad_reviews = count_ngrams(bad_reviews, max_length=3)
    bad_word_cloud_words = return_most_important_words(most_frequent_bad_reviews, 10)
    # Printing the most commonly used 2 and 3 word phrases
    two_phrase_bad_frequent_phrases, three_phrase_bad_frequent_phrases = return_most_frequent_phrases(most_frequent_bad_reviews, num=10)

    # Printing the word cloud and most frequent phrases in the good reviews
    most_frequent_good_reviews = count_ngrams(good_reviews, max_length=3)
    good_word_cloud_words = return_most_important_words(most_frequent_good_reviews, 10)
    two_phrase_good_frequent_phrases, three_phrase_good_frequent_phrases = return_most_frequent_phrases(most_frequent_good_reviews, num=10)


    bwc_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
            ('{business_id}', 'bad_word_cloud', %s)"""
    two_phrase_bad_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
                ('{business_id}', 'bad_2_phrase_cloud', %s)"""
    three_phrase_bad_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
                    ('{business_id}', 'bad_3_phrase_cloud', %s)"""

    gwc_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
                ('{business_id}', 'good_word_cloud', %s)"""
    two_phrase_good_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
                    ('{business_id}', 'good_2_phrase_cloud', %s)"""
    three_phrase_good_sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
                        ('{business_id}', 'good_3_phrase_cloud', %s)"""

    try:
        # create sqlalchemy engine
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user="root",
                                       pw="Cookers00800*",
                                       db="nlp_yelp_reviews"))
        with engine.connect() as con:
            con.execute(bwc_sql, json.dumps(bad_word_cloud_words))
            con.execute(two_phrase_bad_sql, json.dumps(two_phrase_bad_frequent_phrases))
            con.execute(three_phrase_bad_sql, json.dumps(three_phrase_bad_frequent_phrases))
            con.execute(gwc_sql, json.dumps(good_word_cloud_words))
            con.execute(two_phrase_good_sql, json.dumps(two_phrase_good_frequent_phrases))
            con.execute(three_phrase_good_sql, json.dumps(three_phrase_good_frequent_phrases))
    except Exception as e:
        print("Error while saving the data to MySQL", e)
