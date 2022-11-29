from django.db import models


# Create your models here.
class BusinessesData(models.Model):

    business_id = models.CharField(max_length=120)
    business_name = models.CharField(max_length=120)
    stars = models.FloatField()
    review_count = models.IntegerField()
    text = models.TextField()

    class Meta:
        db_table = 'businesses_data'

    __tablename__ = "businesses_data"

class ReviewData(models.Model):
    business_id = models.CharField(max_length=120)
    stars = models.FloatField()
    review_date = models.DateField()
    review_text = models.TextField()
    class Meta:
        db_table = 'review_data'
    __tablename__ = "review_data"


class CommonWordsAndPhrases(models.Model):
    business_id = models.CharField(max_length=120)
    phrase_type = models.TextField()
    phrases = models.TextField()
    class Meta:
        db_table = 'common_words_and_phrases'
    __tablename__ = "common_words_and_phrases"


