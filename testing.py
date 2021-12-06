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
from sklearn.metrics import confusion_matrix
import pickle
# from nltk.corpus import stopwords
nltk.download('stopwords')


nb_filepath = '/home/pes1ug19cs304/Desktop/Project/model_store/2^18/10k/nb.sav'
sgd_filepath = '/home/pes1ug19cs304/Desktop/Project/model_store/2^18/10k/sgd.sav'
pa_filepath = '/home/pes1ug19cs304/Desktop/Project/model_store/2^18/10k/pa.sav'
data_filepath='/home/pes1ug19cs304/Desktop/Project/testing_data.csv'

#create a testing_data.csv with the following as its first line
#batch_no,classifier,accuracy,recall_0,recall_4,precision_0,precision_4,f1_0,f1_4,features,cm00,cm04,cm40,cm44,batch_size

batch_no=1
batch_size=10000 #for the trained model , required for analysis
features = 18 #for the trained model , required for analysis
hv = HashingVectorizer(n_features=2**(features), alternate_sign=False)

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


def train_nb(tweets,y):
    nb_file = open(nb_filepath, 'rb')
    global batch_no
    classifier='NB'
    data_file = open(data_filepath, 'a')
    nb = pickle.load(nb_file)
    pred_y = nb.predict(tweets)
    recall=recall_score(y,pred_y,average=None)
    precision=precision_score(y,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y,pred_y)
    cm = confusion_matrix(y,pred_y)
    global features
    global batch_size
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{features},{cm[0][0]},{cm[0][1]},{cm[1][0]},{cm[1][1]},{batch_size}\n')
    data_file.close()
    print(f'Batch accuracy = {accuracy} - NB')
    print(f'Batch precision = {precision} - NB')
    print(f'Batch recall = {recall} - NB')
    print(f'Batch F1 = {f1} - NB')
    print(f'Confusion Matrix : {cm} - NB')
    nb_file.close()


def train_sgd(tweets,y):
    sgd_file = open(sgd_filepath, 'rb')
    global batch_no
    classifier='SGD'
    data_file = open(data_filepath, 'a')
    sgd = pickle.load(sgd_file)
    pred_y = sgd.predict(tweets)
    recall=recall_score(y,pred_y,average=None)
    precision=precision_score(y,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y,pred_y)
    cm = confusion_matrix(y,pred_y)
    global features
    global batch_size
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{features},{cm[0][0]},{cm[0][1]},{cm[1][0]},{cm[1][1]},{batch_size}\n')
    data_file.close()
    print(f'Batch accuracy = {accuracy} - SGD')
    print(f'Batch precision = {precision} - SGD')
    print(f'Batch recall = {recall} - SGD')
    print(f'Batch F1 = {f1} - SGD')
    print(f'Confusion Matrix : {cm} - SGD')
    sgd_file.close()
    batch_no+=1


def train_pa(tweets,y,):
    pa_file = open(pa_filepath, 'rb')
    pa = pickle.load(pa_file)
    global batch_no
    classifier='PA'
    data_file = open(data_filepath, 'a')
    pred_y = pa.predict(tweets)
    recall=recall_score(y,pred_y,average=None)
    precision=precision_score(y,pred_y,average=None)
    f1=2*recall*precision / (recall+precision)
    accuracy = accuracy_score(y,pred_y)
    cm = confusion_matrix(y,pred_y)
    global features
    global batch_size
    data_file.write(f'{batch_no},{classifier},{accuracy},{recall[0]},{recall[1]},{precision[0]},{precision[1]},{f1[0]},{f1[1]},{features},{cm[0][0]},{cm[0][1]},{cm[1][0]},{cm[1][1]},{batch_size}\n')
    print(batch_no)
    print(f'Batch accuracy = {accuracy} - PA')
    print(f'Batch precision = {precision} - PA')
    print(f'Batch recall = {recall} - PA')
    print(f'Batch F1 = {f1} - PA')
    print(f'Confusion Matrix : {cm} - PA')
    data_file.close()
    pa_file.close()


def process_rdd(rdd):
    df = make_list_json(rdd)
    if df is not None:
        tweets = np.array(df.select('feature1').collect())
        tweets = np.array([preprocess(i[0]) for i in tweets])
        labels = np.array([i[0]
                          for i in list(df.select('feature0').collect())])
        tweets = hv.fit_transform(tweets)
        print('Preprocessing Done')
        train_pa(tweets, labels)
        train_nb(tweets, labels)
        train_sgd(tweets, labels)


sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)

dataStream = ssc.socketTextStream("localhost", 6100)
dataStream.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination(50)
ssc.stop()
