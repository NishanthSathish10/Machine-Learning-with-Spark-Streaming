#! /usr/bin/python3
import string
import re
import json
import numpy as np
import nltk
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.streaming import StreamingContext
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import pickle
# from nltk.corpus import stopwords
nltk.download('stopwords')

hv = HashingVectorizer(n_features=2**18, alternate_sign=False)
nb_filepath = './MLSpark/models/nb.sav'
sgd_filepath = './MLSpark/models/sgd.sav'
pa_filepath = './MLSpark/models/pa.sav'


def make_list_json(rdd):
    taken = rdd.collect()
    if len(taken) > 0:
        taken = taken[0]
        op = json.loads(taken).values()
        df = spark.createDataFrame(Row(**x) for x in op)
        return df


def preprocess(data):
    stop_words = nltk.corpus.stopwords.words('english')

    # split to individual words
    words = data.split()
    # replacing certain characters
    re_char1 = [re.sub("\.", " ", word) for word in words]
    re_char2 = [re.sub("[!?]", "", word) for word in re_char1]
    re_apos = [re.sub("'", "", word) for word in re_char2]
    # convert to lowercase
    lower = [word.lower() for word in re_apos]
    # remove words with punctuation
    re_punc = [word for word in lower if word.isalnum()]
    # remove stop words
    final = [word for word in re_punc if not word in stop_words]
    cleaned = " "
    # replace data with cleaned data
    return cleaned.join(final)


def train_nb(tweets, tweets_test, y, y_test):
    nb_file = open(nb_filepath, 'rb')
    nb = pickle.load(nb_file)
    nb.partial_fit(tweets, y, classes=np.unique(y_test))
    score = nb.score(tweets_test, y_test)
    print(f'Batch accuracy = {score}')
    nb.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    nb_file.close()
    nb_file = open(nb_filepath, 'wb')
    pickle.dump(nb, nb_file)
    nb_file.close()


def train_sgd(tweets, tweets_test, y, y_test):
    sgd_file = open(sgd_filepath, 'rb')
    sgd = pickle.load(sgd_file)
    sgd.partial_fit(tweets, y, classes=np.unique(y_test))
    score = sgd.score(tweets_test, y_test)
    print(f'Batch accuracy = {score}')
    sgd.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    sgd_file.close()
    sgd_file = open(sgd_filepath, 'wb')
    pickle.dump(sgd, sgd_file)
    sgd_file.close()


def train_pa(tweets, tweets_test, y, y_test):
    pa_file = open(pa_filepath, 'rb')
    pa = pickle.load(pa_file)
    pa.partial_fit(tweets, y, classes=np.unique(y_test))
    score = pa.score(tweets_test, y_test)
    print(f'Batch accuracy = {score}')
    pa.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    pa_file.close()
    pa_file = open(nb_filepath, 'wb')
    pickle.dump(pa, pa_file)
    pa_file.close()


def process_rdd(rdd):
    df = make_list_json(rdd)
    if df is not None:
        tweets = np.array(df.select('feature1').collect())
        tweets = np.array([preprocess(i[0]) for i in tweets])
        labels = np.array([i[0]
                          for i in list(df.select('feature0').collect())])
        tweets = hv.fit_transform(tweets)
        tweets, tweets_test, y, y_test = train_test_split(
            tweets, labels, test_size=0.2, random_state=42)
        print('Preprocessing Done')
        train_pa(tweets, tweets_test, y, y_test)


sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)

dataStream = ssc.socketTextStream("localhost", 6100)
dataStream.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination(900)
ssc.stop()
