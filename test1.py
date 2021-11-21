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
# from nltk.corpus import stopwords
nltk.download('stopwords')
  

def make_list_json(rdd):
    taken = rdd.collect()
    if len(taken)>0:
        taken = taken[0]
        op = json.loads(taken).values()
        df = spark.createDataFrame(Row(**x) for x in op)
        return df
        
def preprocess(data):
    stop_words = nltk.corpus.stopwords.words('english')
    #for data in df['feature1']:
    #split to individual words
    words = data.split()
    #remove apostrophe
    re_apos = [re.sub("'", "", word) for word in words]
    #convert to lowercase
    lower = [word.lower() for word in re_apos]
    #remove words with punctuation	
    re_punc = [word for word in lower if word.isalnum()]
    #remove stop words
    final = [word for word in re_punc if not word in stop_words]
    cleaned = " "
    #replace data with cleaned data
    return cleaned.join(final)
    #return df
    	

def process_rdd(rdd):
    df = make_list_json(rdd)
    if df is not None:
        tweets = np.array(df.select('feature1').collect())
        tweets = np.array([preprocess(i[0]) for i in tweets])
        labels = np.array([i[0] for i in list(df.select('feature0').collect())])
        print([tweets, labels])

sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

dataStream=ssc.socketTextStream("localhost",6100)
dataStream.foreachRDD(process_rdd)

ssc.start()
ssc.awaitTermination(120)
ssc.stop()