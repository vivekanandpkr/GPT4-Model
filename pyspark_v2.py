from pyspark.sql import SparkSession
from pyspark.ml.fpm import PrefixSpan
spark = SparkSession.builder.getOrCreate()


print("Done")

# Sample data in correct format
# Each row is a sequence where each transaction is represented as an array of strings
data = [
    [["A"], ["B"], ["C"]],
    [["A"], ["B"], ["D"]],
    [["A"], ["D"], ["B"]],
    [["A"], ["B"], ["D"]],
    [["A"], ["B"], ["F"]]
]

print("Data Loaded")
df = spark.createDataFrame(data, ["sequence"])

print("Model Loaded")
# Use PrefixSpan
ps = PrefixSpan(minSupport=0.3, maxPatternLength=5)
result = ps.findFrequentSequentialPatterns(df)

result.show()
