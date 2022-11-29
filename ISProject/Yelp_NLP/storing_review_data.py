import joblib

from Yelp_NLP.helper_functions import read_file_into_df
# import mysql.connector
# from mysql.connector import Error
import pymysql
from sqlalchemy import create_engine

# Reading the businesses file
yelp_df = read_file_into_df(r"C:\Users\carls\PycharmProjects\ISProject\files\yelp_academic_dataset_review.json")
yelp_df = yelp_df[["business_id", "stars", "text", "date"]]
yelp_df.rename(columns={"text": "review_text", "date": "review_date"}, inplace=True)

try:
    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                           .format(user="root",
                                   pw="Cookers00800*",
                                   db="nlp_yelp_reviews"))

    yelp_df.to_sql('review_data', con=engine, if_exists='append', chunksize=1000, index=False)
except Exception as e:
    print("Error while saving the data to MySQL", e)
