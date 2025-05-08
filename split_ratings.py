from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, row_number
from pyspark.sql.window import Window

def start_spark():
    return SparkSession.builder.appName("Split_Ratings").getOrCreate()

def load_ratings(spark, parquet_path):
    return spark.read.parquet(parquet_path)

# Filter sparse users/movies (optional)
def filter_sparse_users_movies(ratings, min_user_ratings=5, min_movie_ratings=5):
    movie_counts = ratings.groupBy("movieId").count().withColumnRenamed("count", "movie_rating_count") \
                          .filter(col("movie_rating_count") >= min_movie_ratings)
    ratings = ratings.join(movie_counts, "movieId")

    user_counts = ratings.groupBy("userId").count().withColumnRenamed("count", "user_rating_count") \
                         .filter(col("user_rating_count") >= min_user_ratings)
    ratings = ratings.join(user_counts, "userId")
    return ratings.select("userId", "movieId", "rating", "timestamp")

def time_ordered_split(ratings, output_dir):
    # Add row number per user ordered by timestamp
    window_spec = Window.partitionBy("userId").orderBy("timestamp")
    ratings_with_index = ratings.withColumn("row_num", row_number().over(window_spec))

    # Count number of ratings per user
    user_counts = ratings_with_index.groupBy("userId").agg(count("*").alias("total"))

    # Join to get total per user
    ratings_with_total = ratings_with_index.join(user_counts, on="userId")

    # Compute thresholds
    train_frac = 0.6
    val_frac = 0.2
    ratings_split = ratings_with_total.withColumn(
        "split",
        (col("row_num")-1) / col("total")
    )

    train = ratings_split.filter(col("split") <= train_frac).drop("row_num", "total", "split")
    val = ratings_split.filter((col("split") > train_frac) & (col("split") <= train_frac + val_frac)).drop("row_num", "total", "split")
    test = ratings_split.filter(col("split") > train_frac + val_frac).drop("row_num", "total", "split")

    # Save
    train.write.mode("overwrite").parquet(f"{output_dir}/train_ratings.parquet")
    val.write.mode("overwrite").parquet(f"{output_dir}/val_ratings.parquet")
    test.write.mode("overwrite").parquet(f"{output_dir}/test_ratings.parquet")

def main():
    spark = start_spark()
    parquet_path = "hdfs:///user/ml9542_nyu_edu/ratings_full.parquet"
    output_dir = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits"

    ratings = load_ratings(spark, parquet_path)

    # EDA before filtering
    ratings.printSchema()
    total_ratings = ratings.count()
    print(f"Total ratings: {total_ratings}", flush=True)
    num_users = ratings.select("userId").distinct().count()
    num_movies = ratings.select("movieId").distinct().count()
    print(f"Unique users: {num_users}, Unique movies: {num_movies}", flush=True)

    # Filter sparse data
    filtered_ratings = filter_sparse_users_movies(ratings)

    # EDA after filtering
    filtered_ratings.printSchema()
    filtered_count = filtered_ratings.count()
    print(f"Filtered ratings: {filtered_count}", flush=True)
    print(f"Dropped ratings: {total_ratings - filtered_count}", flush=True)
    filtered_users = filtered_ratings.select("userId").distinct().count()
    filtered_movies = filtered_ratings.select("movieId").distinct().count()
    print(f"Remaining users: {filtered_users} (Dropped {num_users - filtered_users})", flush=True)
    print(f"Remaining movies: {filtered_movies} (Dropped {num_movies - filtered_movies})", flush=True)

    # Split chronologically per user
    time_ordered_split(filtered_ratings, output_dir)


    print("SCRIPT COMPLETED SUCCESSFULLY", flush=True)


    spark.stop()

if __name__ == "__main__":
    main()
