import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Used for plotting
import seaborn as sns  # Used for plotting
import json
import string

from nltk.corpus import stopwords
from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from Yelp_NLP.helper_functions import count_ngrams, print_word_cloud, print_most_frequent

# Need to run the below for downloading the stopwords package
# nltk.download('stopwords')

# Reading the file
data_file = open(r"C:\Users\carls\PycharmProjects\ISProject\files\yelp_academic_dataset_review.json", encoding='utf-8')
data = []
for line in data_file:
    data.append(json.loads(line))
yelp_df = pd.DataFrame(data)
data_file.close()

# Getting the different star review
yelp_df_1 = yelp_df[yelp_df['stars'] == 1].head(n=200)
yelp_df_5 = yelp_df[yelp_df['stars'] == 5].head(n=200)

# Merging the above 2
yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])
yelp_df_1_5.info()


################################# IMPORTANT WORDS ###################################
yelp_df['review length'] = yelp_df['text'].apply(len)
sns.set_style('white')

g = sns.FacetGrid(yelp_df, col='stars')
g.map(plt.hist, 'review length')

yelp_bad_reviews = yelp_df[(yelp_df.stars <= 2)]
yelp_good_review = yelp_df[(yelp_df.stars >= 4)]

bad_reviews = yelp_bad_reviews.text
good_reviews = yelp_good_review.text

bad_reviews = bad_reviews.sample(frac=.001, replace=True )
good_reviews = good_reviews.sample(frac=.001, replace=True)

most_frequent_bad_reviews = count_ngrams(bad_reviews,max_length=3)
print_word_cloud(most_frequent_bad_reviews, 10)
print_most_frequent(most_frequent_bad_reviews, num= 10)


most_frequent_good_reviews = count_ngrams(good_reviews, max_length=3)
print_word_cloud(most_frequent_good_reviews, 10)
print_most_frequent(most_frequent_good_reviews, num= 10)


################################# IMPORTANT WORDS ###################################




# printing %
print('1-Star Review Percentage = ', (len(yelp_df_1)/len(yelp_df_1_5))*100, "%")


sns.countplot(y='stars', data=yelp_df_1_5, label='Count')
# sns.countplot(y='stars', data=yelp_df)
# plt.show()


###################################### PRE PROCESSING THE DATA####################################
# Prints all the punctuations
# print(string.punctuation)
Test = "Hello, I am!!"
test_punc_removed = [char for char in Test if char not in string.punctuation]
test_punc_removed_join = ''.join(test_punc_removed)


# Getting rid of the stopwords in English
# print(stopwords.words('english'))
# Getting al the words not in the list of stopwords
test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
# print(test_punc_removed_join_clean)


# Count Vectorizer
sample_data = ["This is the first document", "This is the second document", "Is this the first document?"]
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(sample_data)

# Feature extraction on the sample data
print(vectorizer.get_feature_names())
print(x.toarray())


# Applying the above 3 to our yelp reviews
def message_cleaning(message):
    test_punc_remove = [char for char in message if char not in string.punctuation]
    test_punc_remove_join = ''.join(test_punc_remove)
    test_punc_remove_join_clean = [word for word in test_punc_remove_join.split() if word.lower() not in stopwords.words('english')]
    return test_punc_remove_join_clean

yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)

# Can do cleaning in countervectorized
vectorizer_new = CountVectorizer(analyzer=message_cleaning)

yelp_countvectorizer = vectorizer_new.fit_transform(yelp_df_1_5['text'])
print(yelp_countvectorizer.toarray())
###################################### PRE PROCESSING THE DATA END####################################




# Training the model
NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values

# Creating the model based on the values
NB_classifier.fit(yelp_countvectorizer, label)

# Testing the model to see that it is able to identify the data
testing_sample = ['amazing food! highly recommended']

testing_sample_countvectorize = vectorizer_new.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorize)

print(test_predict)

# 80% for training and 20% for testing
X = yelp_countvectorizer
y = label
# both shape shoudl match
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NB_classifier_train_test = MultinomialNB()
NB_classifier_train_test.fit(X_train, y_train)



# model Evaluation
y_predict_train = NB_classifier_train_test.predict(X_train)

cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)
plt.show()

y_predict_test = NB_classifier_train_test.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test, y_predict_test))

# Add Term Frequency Inverse Document Frequency, is a feature extraction technique, not necessary
# Used as a weighting factor during text search
# from sklearn.feature_extraction.text import TfidfTransformer
# yelp_tfidf = TfidfTransformer().fit_transform(yelp_countvectorizer)
# X = yelp_tfidf
# y = label



