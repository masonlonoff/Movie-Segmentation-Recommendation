from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# Read from HDFS
df = spark.read.parquet("top_100_pairs_large-final.parquet")
df.printSchema()
df.show(100, truncate=False)

jaccard = spark.read.parquet("top_100_pairs_Jaccard.parquet")
jaccard.printSchema()

jaccard.select("jaccard_similarity").summary().show()

user_jaccard = jaccard.select("user1", "user2", "jaccard_similarity")

jaccard.show(100, truncate = False)
user_jaccard.show(100, truncate = False)

user_jaccard.write.csv("user_pairs_similarity.csv", header = True, mode = "overwrite")

