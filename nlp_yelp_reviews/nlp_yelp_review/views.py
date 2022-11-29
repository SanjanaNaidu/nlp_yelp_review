import json
from django.http import JsonResponse

import pandas as pd
from django.shortcuts import render
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
from rest_framework import viewsets
from sqlalchemy import create_engine

from .serializers import NLPYelpReviewSerializer
from .models import BusinessesData, ReviewData, CommonWordsAndPhrases
from django.http import HttpResponse


# Create your views here.
@csrf_exempt
def return_dashboard_stats(request):
    business_id = json.loads(request.body)["location"]["state"]["id"]
    try:
        # create sqlalchemy engine
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user="root",
                                       pw="Cookers00800*",
                                       db="nlp_yelp_reviews"))

        yelp_df = pd.read_sql(sql=f"select phrase_type, phrases from common_words_and_phrases where business_id='{business_id}'", con=engine)
    except Exception as e:
        print("Error while reading the data from MySQL", e)
    bad_word_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "bad_word_cloud"]["phrases"].iloc[0])
    bad_2_phrase_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "bad_2_phrase_cloud"]["phrases"].iloc[0])
    bad_3_phrase_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "bad_3_phrase_cloud"]["phrases"].iloc[0])
    pos_neg_sentiments = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "pos_neg_sentiments"]["phrases"].iloc[0])
    good_word_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "good_word_cloud"]["phrases"].iloc[0])
    good_2_phrase_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "good_2_phrase_cloud"]["phrases"].iloc[0])
    good_3_phrase_cloud = json.loads(yelp_df.loc[yelp_df["phrase_type"] == "good_3_phrase_cloud"]["phrases"].iloc[0])
    return_dict = {"bad_word_cloud": bad_word_cloud, "bad_2_phrase_cloud": bad_2_phrase_cloud, "bad_3_phrase_cloud": bad_3_phrase_cloud,
                   "good_word_cloud": good_word_cloud, "good_2_phrase_cloud": good_2_phrase_cloud, "good_3_phrase_cloud": good_3_phrase_cloud,
                   "pos_neg_sentiments": pos_neg_sentiments}
    return JsonResponse(return_dict)


@csrf_exempt
def return_search_keywords(request):
    search_query = json.loads(request.body)["post_input"]["searchInput"]
    business_id = json.loads(request.body)["post_input"]["business_id"]
    try:
        # create sqlalchemy engine
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                               .format(user="root",
                                       pw="Cookers00800*",
                                       db="nlp_yelp_reviews"))
        sql = f"""select stars, review_date, review_text from review_data where business_id='{business_id}' and review_text like %s"""
        args = ['%' + search_query + '%']
        yelp_df = pd.read_sql(sql=sql, params=args, con=engine)
    except Exception as e:
        print("Error while reading the data from MySQL", e)
    return_list = [[yelp_df["stars"].iloc[i],yelp_df["review_date"].iloc[i].strftime("%m/%d/%Y"), yelp_df["review_text"].iloc[i]] for i in range(0, len(yelp_df))]
    return_list.insert(0, ["Stars", "Review Date", "Review Text"])
    final_string = ""
    for i in return_list:
        final_string = final_string + str(i[0]) + "," + i[1] + ',"' + i[2] + '"' + "\n"
    # return_dict = {i: {"stars": yelp_df["stars"].iloc[i], "review_date": yelp_df["review_date"].iloc[i], "review_text": yelp_df["review_text"].iloc[i]} for i in range(0, len(yelp_df))}

    return JsonResponse(final_string, safe=False)


class NLPYelpReviewView(viewsets.ModelViewSet):
    # try:
    #     # create sqlalchemy engine
    #     engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
    #                            .format(user="root",
    #                                    pw="Cookers00800*",
    #                                    db="nlp_yelp_reviews"))
    #     yelp_df = pandas.read_sql(sql="select business_id, stars, review_date, review_text from review_data",
    #                               con=engine)

    serializer_class = NLPYelpReviewSerializer
    # queryset = BusinessesData.objects.all()
    # print(queryset)
