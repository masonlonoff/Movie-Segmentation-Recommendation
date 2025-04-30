from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Read from HDFS (path must match the Hadoop fs)
df = spark.read.parquet("top_100_pairs_large-3.parquet")
df.printSchema()
df.show(100, truncate=False)


