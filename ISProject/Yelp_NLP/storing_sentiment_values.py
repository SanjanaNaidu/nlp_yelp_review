import json
import sys

import pandas

from sqlalchemy import create_engine

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
    yelp_sentiment_df = yelp_df.loc[(yelp_df.stars != 3) & (yelp_df.business_id == business_id)]
    yelp_sentiment_df['sentiment'] = yelp_sentiment_df['stars'].apply(lambda rating: +1 if rating > 3 else -1)

    positive = yelp_sentiment_df[yelp_sentiment_df['sentiment'] == 1]
    negative = yelp_sentiment_df[yelp_sentiment_df['sentiment'] == -1]

    yelp_sentiment_df['sentiment_new'] = yelp_sentiment_df['sentiment'].replace({-1: 'negative'})
    yelp_sentiment_df['sentiment_new'] = yelp_sentiment_df['sentiment_new'].replace({1: 'positive'})
    num_positive = yelp_sentiment_df["sentiment_new"].value_counts()[0]
    num_negative = yelp_sentiment_df["sentiment_new"].value_counts()[1]

    sql = f"""INSERT INTO common_words_and_phrases (business_id, phrase_type, phrases) VALUES
            ('{business_id}', 'pos_neg_sentiments', %s)"""

    try:
        # create sqlalchemy engine
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user="root",
                                       pw="Cookers00800*",
                                       db="nlp_yelp_reviews"))
        with engine.connect() as con:
            con.execute(sql, json.dumps([int(num_positive), int(num_negative)]))
    except Exception as e:
        print("Error while saving the data to MySQL", e)
