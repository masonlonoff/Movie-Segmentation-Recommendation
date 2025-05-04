from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# Read from HDFS (path must match the Hadoop fs)
df = spark.read.parquet("top_100_pairs_large-final.parquet")
df.printSchema()
df.show(100, truncate=False)

jaccard = spark.read.parquet("top_100_pairs_Jaccard.parquet")
jaccard.printSchema()

jaccard.select("jaccard_similarity").summary().show()
#pair_count = jaccard.select("user1", "user2").distinct().count()
#print(f"Number of distinct user pairs: {pair_count}")

#jaccard.orderBy(col("jaccard_similarity").desc()).show(5, truncate=False)
#jaccard.orderBy(col("jaccard_similarity").asc()).show(5, truncate=False)

#jaccard.select("jaccard_similarity").agg({"jaccard_similarity": "min"}).show()
#jaccard.select("jaccard_similarity").agg({"jaccard_similarity": "max"}).show()
user_jaccard = jaccard.select("user1", "user2", "jaccard_similarity")

jaccard.show(100, truncate = False)
user_jaccard.show(100, truncate = False)

user_jaccard.write.csv("user_pairs_similarity.csv", header = True, mode = "overwrite")

