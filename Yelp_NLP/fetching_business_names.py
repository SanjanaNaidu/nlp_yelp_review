from Yelp_NLP.helper_functions import read_file_into_df
# import mysql.connector
# from mysql.connector import Error
import pymysql
from sqlalchemy import create_engine

# Reading the businesses file
businesses_df = read_file_into_df(r"C:\Users\carls\PycharmProjects\ISProject\files\yelp_academic_dataset_business.json")
businesses_df = businesses_df[["business_id", "name", "stars", "review_count", "categories"]]
businesses_df.rename(columns={"name": "business_name"}, inplace=True)

try:
    # create sqlalchemy engine
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                           .format(user="root",
                                   pw="Cookers00800*",
                                   db="nlp_yelp_reviews"))

    businesses_df.to_sql('businesses_data', con=engine, if_exists='append', chunksize=1000, index=False)
except Exception as e:
    print("Error while saving the data to MySQL", e)


# try:
#     connection = mysql.connector.connect(host='localhost',
#                                          database='nlp_yelp_reviews',
#                                          user='root',
#                                          password='Cookers00800*')
#     if connection.is_connected():
#         db_Info = connection.get_server_info()
#         print("Connected to MySQL Server version ", db_Info)
#         cursor = connection.cursor()
#         cursor.execute("select database();")
#         record = cursor.fetchone()
#         print("You're connected to database: ", record)
#
# except Error as e:
#     print("Error while connecting to MySQL", e)
# finally:
#     if connection.is_connected():
#         cursor.close()
#         connection.close()
#         print("MySQL connection is closed")





