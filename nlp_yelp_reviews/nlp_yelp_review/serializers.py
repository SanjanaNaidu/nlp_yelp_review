from rest_framework import serializers
from .models import BusinessesData, ReviewData, CommonWordsAndPhrases


class NLPYelpReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = BusinessesData
        fields = ('business_id', 'business_name', 'stars', 'review_count', 'text')
