import collections
import json
import re

import pandas as pd

from pandas import DataFrame

import pandas as pd
import matplotlib.pyplot as plt  # Used for plotting
import json

from nltk.corpus import stopwords
from wordcloud import WordCloud



def read_file_into_df(file_name: str) -> DataFrame:
    """
    Reading the file specified into a DF and returning it
    :param file_name: Name of the file
    :return: DataFrame with the read file data
    """
    data_file = open(file_name, encoding='utf-8')
    data = []
    for line in data_file:
        data.append(json.loads(line))
    read_df = pd.DataFrame(data)
    data_file.close()
    return read_df


def tokenize(s):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    word_list = re.findall(r'\w+', s.lower())
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    return filtered_words


def count_ngrams(lines, min_length=2, max_length=4):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()
    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()


    return ngrams

def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-word phrase -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')


def return_most_frequent_phrases(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    two_word_phrases = {}
    three_word_phrases = {}

    for n in sorted(ngrams):
        if n == 2:
            for gram, count in ngrams[n].most_common(num):
                two_word_phrases[' '.join(gram)] = count
        elif n == 3:
            for gram, count in ngrams[n].most_common(num):
                three_word_phrases[' '.join(gram)] = count

    return two_word_phrases, three_word_phrases

def print_word_cloud(ngrams, num=5):
    """Print word cloud image plot """
    words = []
    for n in sorted(ngrams):
        for gram, count in ngrams[n].most_common(num):
            s = ' '.join(gram)
            words.append(s)

    cloud = WordCloud(width=1440, height=1080, max_words=200).generate(' '.join(words))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off')
    plt.show()
    print('')


def return_most_important_words(ngrams, num=5):
    words = []
    for n in sorted(ngrams):
        for gram, count in ngrams[n].most_common(num):
            s = ' '.join(gram)
            words.append(s)
    return words