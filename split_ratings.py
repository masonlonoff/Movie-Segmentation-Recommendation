from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col

def start_spark():
    spark = SparkSession.builder.appName("Split_Ratings").getOrCreate()
    return spark

def load_ratings(spark, parquet_path):
    ratings = spark.read.parquet(parquet_path)
    return ratings

# Filter sparse users/movies (optional)
def filter_sparse_users_movies(ratings, min_user_ratings=5, min_movie_ratings=5):
    movie_counts = ratings.groupBy("movieId").count().withColumnRenamed("count", "movie_rating_count") \
            .filter(col("movie_rating_count") >= min_movie_ratings)
    ratings = ratings.join(movie_counts, "movieId")
    
    user_counts = ratings.groupBy("userId").count().withColumnRenamed("count", "user_rating_count") \
            .filter(col("user_rating_count") >= min_user_ratings)
    ratings = ratings.join(user_counts, "userId")

    return ratings

def split_and_save(ratings, output_dir):
    train, val, test = ratings.randomSplit([0.6, 0.2, 0.2], seed = 23)
    
    print("Train count:", train.count())
    print("Validation count:", val.count())
    print("Test count:", test.count())

    train.write.mode("overwrite").parquet(f"{output_dir}/train_ratings.parquet")
    val.write.mode("overwrite").parquet(f"{output_dir}/val_ratings.parquet")
    test.write.mode("overwrite").parquet(f"{output_dir}/test_ratings.parquet")

    print(f"Splits saved to: {output_dir}", flush = True)

def main():
    spark = start_spark()

    parquet_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.parquet"
    output_dir = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits"

    ratings = load_ratings(spark, parquet_path)

    # EDA BEFORE filtering
    total_ratings = ratings.count()
    print(f"Total ratings: {total_ratings}")
    num_users = ratings.select("userId").distinct().count()
    num_movies = ratings.select("movieId").distinct().count()
    print(f"Unique users: {num_users}, Unique movies: {num_movies}")

    # Apply filtering BEFORE splitting
    filtered_ratings = filter_sparse_users_movies(ratings)

    # EDA AFTER filtering
    filtered_count = filtered_ratings.count()
    print(f"Filtered ratings: {filtered_count}")
    print(f"Dropped ratings: {total_ratings - filtered_count}")
    filtered_users = filtered_ratings.select("userId").distinct().count()
    filtered_movies = filtered_ratings.select("movieId").distinct().count()
    print(f"Remaining users: {filtered_users} (Dropped {num_users - filtered_users})")
    print(f"Remaining movies: {filtered_movies} (Dropped {num_movies - filtered_movies})")

    # Now split and save the FILTERED data
    print("saving file")
    split_and_save(filtered_ratings, output_dir)

    spark.stop()


if __name__ == "__main__":
    main()
