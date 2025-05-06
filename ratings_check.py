from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder.appName("Inspect Ratings Full").getOrCreate()
    
    # Adjust this path if needed
    path = "hdfs:///user/ml9542_nyu_edu/ratings_full.parquet"
    df = spark.read.parquet(path)

    # Show schema and a few rows
    df.printSchema()
    df.show(10, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()

