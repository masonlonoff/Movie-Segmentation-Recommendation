# eda_small.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, explode, split

# Start Spark Session
spark = SparkSession.builder.appName("Capstone_EDA_Small").getOrCreate()

# Load ratings.csv
ratings = spark.read.csv("hdfs:///user/ml9542_nyu_edu/ml-latest-small/ratings.csv", header=True, inferSchema=True)

# Load movies.csv
movies = spark.read.csv("hdfs:///user/ml9542_nyu_edu/ml-latest-small/movies.csv", header=True, inferSchema=True)

# Load tags.csv
tags = spark.read.csv("hdfs:///user/ml9542_nyu_edu/ml-latest-small/tags.csv", header=True, inferSchema=True)

# Load links.csv
links = spark.read.csv("hdfs:///user/ml9542_nyu_edu/ml-latest-small/links.csv", header=True, inferSchema=True)

# ===== Ratings Overview =====
print("\n===== Ratings Schema =====")
ratings.printSchema()
print("\n===== Ratings Sample =====")
ratings.show(5)
print("\n===== Ratings Count =====")
print(ratings.count())

# ===== Movies Overview =====
print("\n===== Movies Schema =====")
movies.printSchema()
print("\n===== Movies Sample =====")
movies.show(5)
print("\n===== Movies Count =====")
print(movies.count())

# ===== Tags Overview =====
print("\n===== Tags Schema =====")
tags.printSchema()
print("\n===== Tags Sample =====")
tags.show(5)
print("\n===== Tags Count =====")
print(tags.count())

# ===== Links Overview =====
print("\n===== Links Schema =====")
links.printSchema()
print("\n===== Links Sample =====")
links.show(5)
print("\n===== Links Count =====")
print(links.count())

# ===== Null Checking =====
def check_nulls(df, name):
    print(f"\n===== Null check for {name} =====")
    df.select([_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()

check_nulls(ratings, "Ratings")
check_nulls(movies, "Movies")
check_nulls(tags, "Tags")
check_nulls(links, "Links")

# ===== Ratings Distribution =====
print("\n===== Ratings Distribution =====")
ratings.groupBy("rating").count().orderBy("rating").show()

# ===== Genre Distribution =====
print("\n===== Genre Distribution =====")
movies.withColumn("genre", explode(split("genres", "\\|"))) \
      .groupBy("genre").count().orderBy("count", ascending=False).show()

spark.stop()

