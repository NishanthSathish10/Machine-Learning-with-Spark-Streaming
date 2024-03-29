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
km_path = '/home/pes1ug19cs304/Desktop/Project/km.sav' 

batch_no =1
batch_size=5000

def cluster(tweets):
	global batch_no
	km_file = open(km_path, 'rb')
	km = pickle.load(km_file)
	km_file.close()
	km.partial_fit(tweets)
	print(f'Batch number : {batch_no}')
	print(km.cluster_centers_)
	km_file = open(km_path,'wb')
	pickle.dump(km,km_file)
	km_file.close()
	batch_no += 1

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
    
def process_rdd(rdd):
    df = make_list_json(rdd)
    if df is not None:
        tweets = np.array(df.select('feature1').collect())
        tweets = np.array([preprocess(i[0]) for i in tweets])
        labels = np.array([i[0]
                          for i in list(df.select('feature0').collect())])
        tweets = hv.fit_transform(tweets)
        print('Preprocessing Done')
        cluster(tweets)
        


sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)

dataStream = ssc.socketTextStream("localhost", 6100)
dataStream.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination(100)
ssc.stop()
