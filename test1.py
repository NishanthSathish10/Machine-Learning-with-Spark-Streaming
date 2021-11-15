from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.streaming import StreamingContext

sc = SparkContext("local[2]", "name")
spark = SparkSession(sc)
ssc = StreamingContext(sc, 1)
sql_context=SQLContext(sc)

dataStream=ssc.socketTextStream("localhost",6100)
dataStream.pprint()

ssc.start()
ssc.awaitTermination()
ssc.stop()

