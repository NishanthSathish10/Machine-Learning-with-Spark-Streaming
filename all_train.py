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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import pickle
# from nltk.corpus import stopwords
nltk.download('stopwords')

hv = HashingVectorizer(n_features=2**18, alternate_sign=False)
nb_filepath = '/home/pes1ug19cs304/Desktop/Project/models/nb.sav'
sgd_filepath = '/home/pes1ug19cs304/Desktop/Project/models/sgd.sav'
pa_filepath = '/home/pes1ug19cs304/Desktop/Project/models/pa.sav'
data_filepath='/home/pes1ug19cs304/Desktop/Project/data/data218.csv'

#create a csv file with the following as the first line for data collection
#batch_no,classifier,accuracy,recall_0,recall_4,precision_0,precision_4,f1_0,f1_4,batch_size

batch_no=1
batch_size=10000

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
    global batch_no
    classifier='NB'
    data_file = open(data_filepath, 'a')
    nb = pickle.load(nb_file)
    nb.partial_fit(tweets, y, classes=np.unique(y_test))
    #score = nb.score(tweets_test, y_test)
    pred_y = nb.predict(tweets_test)
    #print(pred_y)
    #print(y_test)
    #print(f'Batch accuracy = {score} - NB')
    recall=recall_score(y_test,pred_y,average=None)
    precision=precision_score(y_test,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y_test,pred_y)
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{batch_size}\n')
    data_file.close()
    #print(batch_no)
    #batch_no+=1
    #print(f'Batch accuracy = {accuracy} - NB')
    #print(f'Batch precision = {precision} - NB')
    #print(f'Batch recall = {recall} - NB')
    #print(f'Batch F1 = {f1} - NB')
    nb.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    nb_file.close()
    nb_file = open(nb_filepath, 'wb')
    pickle.dump(nb, nb_file)
    nb_file.close()


def train_sgd(tweets, tweets_test, y, y_test):
    sgd_file = open(sgd_filepath, 'rb')
    global batch_no
    classifier='SGD'
    data_file = open(data_filepath, 'a')
    sgd = pickle.load(sgd_file)
    sgd.partial_fit(tweets, y, classes=np.unique(y_test))
    pred_y = sgd.predict(tweets_test)
    recall=recall_score(y_test,pred_y,average=None)
    precision=precision_score(y_test,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y_test,pred_y)
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{batch_size}\n')
    data_file.close()
    #score = sgd.score(tweets_test, y_test)
    #print(f'Batch accuracy = {score} - SGD')
    sgd.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    sgd_file.close()
    batch_no+=1
    sgd_file = open(sgd_filepath, 'wb')
    pickle.dump(sgd, sgd_file)
    sgd_file.close()


def train_pa(tweets, tweets_test, y, y_test):
    pa_file = open(pa_filepath, 'rb')
    pa = pickle.load(pa_file)
    global batch_no
    classifier='PA'
    data_file = open(data_filepath, 'a')
    pa.partial_fit(tweets, y, classes=np.unique(y_test))
    #score = pa.score(tweets_test, y_test)
    #print(f'Batch accuracy = {score} - PA')
    pred_y = pa.predict(tweets_test)
    recall=recall_score(y_test,pred_y,average=None)
    precision=precision_score(y_test,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y_test,pred_y)
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{batch_size}\n')
    data_file.close()
    pa.partial_fit(tweets_test, y_test, classes=np.unique(
        y_test))  # cross validation lol
    pa_file.close()
    pa_file = open(pa_filepath, 'wb')
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
        train_nb(tweets, tweets_test, y, y_test)
        train_sgd(tweets, tweets_test, y, y_test)


sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)

dataStream = ssc.socketTextStream("localhost", 6100)
dataStream.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination(int((1520000/batch_size)*5)+60)
ssc.stop()
