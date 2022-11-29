import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Used for plotting
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
import seaborn as sns  # Used for plotting
import json
import string

from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from Yelp_NLP.helper_functions import count_ngrams, print_word_cloud, print_most_frequent, read_file_into_df
import joblib
# Need to run the below for downloading the stopwords package
# nltk.download('stopwords')

# Reading the file
yelp_df = read_file_into_df(r"C:\Users\carls\PycharmProjects\ISProject\files\yelp_academic_dataset_review.json")

###################### SENTIMENT ANALYSIS #####################

yelp_df = yelp_df.head(1000)
yelp_sentiment_df = yelp_df[yelp_df['stars'] != 3]
yelp_sentiment_df['sentiment'] = yelp_sentiment_df['stars'].apply(lambda rating : +1 if rating > 3 else -1)

positive = yelp_sentiment_df[yelp_sentiment_df['sentiment'] == 1]
negative = yelp_sentiment_df[yelp_sentiment_df['sentiment'] == -1]

yelp_sentiment_df['sentiment_new'] = yelp_sentiment_df['sentiment'].replace({-1 : 'negative'})
yelp_sentiment_df['sentiment_new'] = yelp_sentiment_df['sentiment_new'].replace({1 : 'positive'})
fig = px.histogram(yelp_sentiment_df, x="sentiment_new")
fig.update_traces(marker_color="indianred", marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


###################### SENTIMENT ANALYSIS #####################




# Can filter by business ID here
business_id = "SZU9c8V2GuREDN5KgyHFJw"

################################# IMPORTANT WORDS ###################################
yelp_df['review length'] = yelp_df['text'].apply(len)
sns.set_style('white')

# Visualizes the text lengths based on the stars
g = sns.FacetGrid(yelp_df, col='stars')
g.map(plt.hist, 'review length')

# Filtering based on business ID
yelp_bad_reviews = yelp_df.loc[(yelp_df.stars <= 2) & (yelp_df.business_id == business_id)]
yelp_good_review = yelp_df.loc[(yelp_df.stars >= 4) & (yelp_df.business_id == business_id)]

bad_reviews = yelp_bad_reviews.text
good_reviews = yelp_good_review.text

# Taking a sample set from the bad and good reviews
bad_reviews = bad_reviews.sample(frac=.001, replace=True)
good_reviews = good_reviews.sample(frac=.001, replace=True)

# Printing the word cloud and most frequent phrases in the bad reviews
most_frequent_bad_reviews = count_ngrams(bad_reviews,max_length=3)
print_word_cloud(most_frequent_bad_reviews, 10)
# Printing the most commonly used 2 and 3 word phrases
print_most_frequent(most_frequent_bad_reviews, num=10)

# Printing the word cloud and most frequent phrases in the good reviews
most_frequent_good_reviews = count_ngrams(good_reviews, max_length=3)
print_word_cloud(most_frequent_good_reviews, 10)
print_most_frequent(most_frequent_good_reviews, num=10)


################################# IMPORTANT WORDS END ###################################


# Getting the different star review
yelp_df_1 = yelp_df[yelp_df['stars'] == 1]
yelp_df_5 = yelp_df[yelp_df['stars'] == 5]
yelp_df_5 = yelp_df_5[["stars", "text"]]
yelp_df_1 = yelp_df_1[["stars", "text"]]
# Merging the above 2
yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])
yelp_df_1_5.info()


# printing %
print('1-Star Review Percentage = ', (len(yelp_df_1)/len(yelp_df_1_5))*100, "%")


sns.countplot(y='stars', data=yelp_df_1_5, label='Count')
sns.countplot(y='stars', data=yelp_df)
plt.show()


###################################### PRE PROCESSING THE DATA####################################
count = 0
def message_cleaning(message):
    punc_remove = [char for char in message if char not in string.punctuation]
    punc_remove_join = ''.join(punc_remove)
    punc_remove_join_clean = [word for word in punc_remove_join.split() if
                              word.lower() not in stopwords.words('english')]
    return punc_remove_join_clean


# yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)
# yelp_df_1_5['text'] = yelp_df_1_5['text'].apply(message_cleaning)

# Can do cleaning in countervectorized
vectorizer_new = CountVectorizer(analyzer=message_cleaning)

yelp_countvectorizer = vectorizer_new.fit_transform(yelp_df_1_5['text'])

# print(yelp_countvectorizer.toarray())
print("Here")
###################################### PRE PROCESSING THE DATA END####################################


###################################### CREATING THE MULTINOMIALNB MODEL####################################
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
# plt.show()

y_predict_test = NB_classifier_train_test.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
# plt.show()

print(classification_report(y_test, y_predict_test))

# Dumping the created model into a pickle file for use later
joblib.dump(NB_classifier, 'NB_classifier_model.pkl')

###################################### CREATING THE MULTINOMIALNB MODEL END####################################

############################### CREATING THE SENTIMENT ANALYSIS USING A LOGISTIC REGRESSION MODEL ###################

def sentiment_message_cleaning(message):
    punc_remove = [char for char in message if char not in string.punctuation]
    punc_remove_join = ''.join(punc_remove)
    punc_remove_join_clean = " ".join([word for word in punc_remove_join.split() if word.lower() not in stopwords.words('english')])
    return punc_remove_join_clean

yelp_df_sentiment = yelp_df[yelp_df['stars'] != 3]
yelp_df_sentiment['sentiment'] = yelp_df_sentiment['stars'].apply(lambda rating : +1 if rating > 3 else -1)
yelp_df_sentiment['text'] = yelp_df_sentiment['text'].apply(sentiment_message_cleaning)

index = yelp_df_sentiment.index
yelp_df_sentiment['random_number'] = np.random.randn(len(index))
train = yelp_df_sentiment[yelp_df_sentiment['random_number'] <= 0.8]
test = yelp_df_sentiment[yelp_df_sentiment['random_number'] > 0.8]

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train['text'])
test_matrix = vectorizer.transform(test['text'])

lr = LogisticRegression(solver='lbfgs', max_iter=6000)

X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']


scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr.fit(X_train, y_train)

predictions = lr.predict(X_test)
print(predictions)

print(classification_report(predictions,y_test))

# Dumping the created model into a pickle file for use later
joblib.dump(lr, 'Sentiment_classifier_model.pkl')
############################### CREATING THE SENTIMENT ANALYSIS USING A LOGISTIC REGRESSION MODEL END ##################