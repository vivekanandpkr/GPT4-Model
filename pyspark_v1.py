from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.fpm import PrefixSpan

# build a SparkConf object that contains information about your application
conf = SparkConf().setAppName("shell").setMaster("local[*]")  # app name and master
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Assume you have sequences in a format such as:
sequences = [[['A'], ['B'], ['C']], [['A'], ['B'], ['D']], [['E'], ['F']]]

# Convert your data to an RDD
rdd = sc.parallelize(sequences, numSlices=10)

# Convert your RDD to a DataFrame
df = spark.createDataFrame(rdd, ["sequence"])

# Initialize PrefixSpan model
prefixSpan = PrefixSpan(minSupport=0.5, maxPatternLength=5)

# Frequent sequences
frequent_seq_df = prefixSpan.findFrequentSequentialPatterns(df)

# To view the result
frequent_seq_df.show()
