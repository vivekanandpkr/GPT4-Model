from pyspark.sql import SparkSession
from pyspark.ml.fpm import PrefixSpan
from pyspark.sql.functions import col, explode, array, collect_list


spark = SparkSession.builder \
    .appName("shell") \
    .getOrCreate()

# Original data
data = [
    ("1", ["A", "B", "C"]),
    ("2", ["A", "B", "D"]),
    ("3", ["A", "D", "E"]),
    ("4", ["A", "B", "F"]),
]

# Convert to dataframe
df = spark.createDataFrame(data, ["id", "sequence"])

# Transform sequence to array of arrays
from pyspark.sql.functions import col, explode, array
df = df.withColumn("sequence", explode(col("sequence"))) \
        .groupBy("id") \
        .agg(collect_list(array(col("sequence"))).alias("sequence"))

df.show()

ps = PrefixSpan(minSupport=0.3, maxPatternLength=5)
result = ps.findFrequentSequentialPatterns(df)
result.show()
