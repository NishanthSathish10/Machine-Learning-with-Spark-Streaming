from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext
import json
def make_list_json(rdd):
    taken = rdd.collect()
    if len(taken)>0:
        taken = taken[0]
        op = json.loads(taken).values()
        print(list(op))

sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

dataStream=ssc.socketTextStream("localhost",6100)
dataStream.foreachRDD(make_list_json)

ssc.start()
ssc.awaitTermination(60)
ssc.stop()

