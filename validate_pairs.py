from pyspark.sql import SparkSession
from pyspark.sql.functions import col, corr, count, avg, stddev, min, max, rand, countDistinct, abs

def start_spark():
    spark = SparkSession.builder.appName("Validate_Movie_Twins").getOrCreate()
    return spark

def main():
    spark = start_spark()

    # Load full ratings data and top 100 pairs
    ratings = spark.read.parquet("ratings_full.parquet")
    top_pairs = spark.read.parquet("top_100_pairs_large-final.parquet")

    # Prep ratings for joining
    ratings1 = ratings.selectExpr("userId as user1", "movieId", "rating as rating1")
    ratings2 = ratings.selectExpr("userId as user2", "movieId", "rating as rating2")

    # Join top pairs with ratings and compute correlations
    joined_top = top_pairs.join(ratings1, on="user1") \
                          .join(ratings2, on=["user2", "movieId"])

    grouped_top = joined_top.groupBy("user1", "user2") \
                            .agg(
                                corr("rating1", "rating2").alias("rating_correlation"),
                                count("*").alias("overlap_count")
                            )

    grouped_top_valid = grouped_top.filter(col("rating_correlation").isNotNull())

    # EDA summary for top pairs
    print("\n=== EDA: Pair Correlation Summary ===")
    grouped_top_valid.selectExpr(
        "count(*) as num_pairs",
        "avg(rating_correlation) as avg_corr",
        "stddev(rating_correlation) as stddev_corr",
        "min(rating_correlation) as min_corr",
        "max(rating_correlation) as max_corr",
        "avg(overlap_count) as avg_overlap_movies"
    ).show()

    # === RANDOM PAIR VALIDATION ===

    # Sample random users
    all_users = ratings.select("userId").distinct()
    sampled_users = all_users.orderBy(rand(seed=7))

    # Form 100 random user pairs
    user_pairs_random = sampled_users.alias("a").crossJoin(sampled_users.alias("b")) \
        .filter("a.userId < b.userId") \
        .select(col("a.userId").alias("user1"), col("b.userId").alias("user2")) \
        .limit(10000)

    # Join random pairs with ratings
    joined_rand = user_pairs_random.join(ratings1, on="user1") \
                                   .join(ratings2, on=["user2", "movieId"])

    grouped_rand = joined_rand.groupBy("user1", "user2") \
                              .agg(
                                  corr("rating1", "rating2").alias("rating_correlation"),
                                  count("*").alias("overlap_count")
                              )

    grouped_rand_valid = grouped_rand.filter(col("rating_correlation").isNotNull()) \
                                     .orderBy(rand()).limit(100)

    # EDA summary for random pairs
    print("\n=== Random Pair Correlation Summary ===")
    grouped_rand_valid.selectExpr(
        "count(*) as num_pairs",
        "avg(rating_correlation) as avg_corr",
        "stddev(rating_correlation) as stddev_corr",
        "min(rating_correlation) as min_corr",
        "max(rating_correlation) as max_corr",
        "avg(overlap_count) as avg_overlap_movies"
    ).show()


    # === DROPPED PAIR ANALYSIS ===

    # Count overlapping movies per pair
    overlap_counts = joined_top.groupBy("user1", "user2").agg(count("movieId").alias("overlap_count"))

    # Attach overlap to valid pairs
    valid_pairs_with_overlap = grouped_top_valid.join(overlap_counts, on=["user1", "user2"])

    # Attach overlap to all top pairs (some will have nulls for missing join)
    all_pairs_with_overlap = top_pairs.join(overlap_counts, on=["user1", "user2"], how="left")

    # Find dropped pairs
    dropped_pairs = all_pairs_with_overlap.join(valid_pairs_with_overlap.select("user1", "user2"),
                                                on=["user1", "user2"],
                                                how="left_anti")

    print("\n=== Dropped Pair Overlap Stats ===")
    dropped_pairs.selectExpr(
        "count(*) as num_dropped_pairs",
        "avg(overlap_count) as avg_overlap_dropped",
        "min(overlap_count) as min_overlap_dropped",
        "max(overlap_count) as max_overlap_dropped",
        "stddev(overlap_count) as stddev_overlap_dropped"
    ).show()

    # Join dropped pairs with their ratings
    joined_dropped = dropped_pairs.join(ratings1, on="user1") \
                              .join(ratings2, on=["user2", "movieId"])

    # Compute stddev of ratings for each user in the pair (on shared movies only)
    stddev_check = joined_dropped.groupBy("user1", "user2") \
        .agg(
            countDistinct("movieId").alias("num_shared_movies"),
            stddev("rating1").alias("stddev_user1"),
            stddev("rating2").alias("stddev_user2")
        )

# Show where stddev is null or zero (no variance = correlation undefined)
    stddev_check.filter(
        (col("stddev_user1").isNull()) | (col("stddev_user2").isNull()) |
        (col("stddev_user1") == 0) | (col("stddev_user2") == 0)
    ).show()

# Compute average rating per user (on shared movies only)
    avg_ratings = joined_dropped.groupBy("user1", "user2") \
        .agg(
            avg("rating1").alias("avg_rating_user1"),
            avg("rating2").alias("avg_rating_user2")
        )

# Compute absolute difference in average ratings
    avg_ratings = avg_ratings.withColumn(
        "rating_diff", abs(col("avg_rating_user1") - col("avg_rating_user2"))
    )

# Show summary statistics for rating differences
    print("\n=== Average Rating Comparison for Dropped Pairs ===")
# Show full average rating comparison for each dropped pair
    avg_ratings.select(
        "user1",
        "user2",
        "avg_rating_user1",
        "avg_rating_user2",
        "rating_diff"
    ).orderBy("rating_diff").show(50, truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()

