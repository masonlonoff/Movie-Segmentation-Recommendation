from pyspark.sql import SparkSession
import random   
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from pyspark.sql.functions import collect_set, collect_list, col, explode, udf, array_intersect, size
from pyspark.sql.types import ArrayType, IntegerType
import numpy as np


#random.seed(11)


# --- Step 0: Initialize Spark ---
def start_spark():
    spark = SparkSession.builder.appName("Capstone_MinHash").getOrCreate()
    return spark

# --- Step 1: Load Ratings Data ---


def load_ratings(spark, parquet_path, csv_path):
    # Try to read Parquet first
    try:
        ratings = spark.read.parquet(parquet_path)
        print("Successfully loaded from Parquet file.")
    except Exception as e:
        print("Parquet file not found. Reading from CSV and converting to Parquet...")
        ratings = spark.read.csv(csv_path, header=True, inferSchema=True)
        ratings = ratings.select("userId", "movieId").dropna()
        ratings.write.mode("overwrite").parquet(parquet_path)
        print("Parquet file created successfully.")
    return ratings

# --- Step 2: Build User Movie Sets ---
def build_user_movie_sets(ratings):
    user_movies = ratings.groupBy("userId").agg(collect_set("movieId").alias("movieSet"))
    return user_movies

# --- Step 3: Generate MinHash Signatures ---
def generate_minhash_signatures(user_movies, num_hashes=128):
    np.random.seed(11)

    # Use large prime and coefficient range
    prime = 10**9 + 7
    max_val = prime - 1

    a_vals = np.random.randint(1, max_val, size=num_hashes).tolist()
    b_vals = np.random.randint(0, max_val, size=num_hashes).tolist()

    def make_minhash_udf(a_vals, b_vals, prime):
        def minhash_movie_set(movie_set):
            if movie_set is None or len(movie_set) == 0:
                return [prime] * num_hashes
            signature = []
            for i in range(num_hashes):
                min_hash = min([(a_vals[i] * m + b_vals[i]) % prime for m in movie_set])
                signature.append(min_hash)
            return signature
        return udf(minhash_movie_set, ArrayType(IntegerType()))

    minhash_udf = make_minhash_udf(a_vals, b_vals, prime)
    user_signatures = user_movies.withColumn("minhashSignature", minhash_udf("movieSet"))

    return user_signatures
def apply_lsh(user_signatures, num_bands=32, rows_per_band=4):
    # Step 1: Split MinHash signatures into bands
    def split_into_bands(signature):
        bands = []
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band = signature[start:end]
            bands.append((band_idx, tuple(band)))
        return bands

    # Define the UDF to split signatures into bands
    split_udf = udf(split_into_bands, ArrayType(StructType([
        StructField("band_idx", IntegerType()),
        StructField("band", ArrayType(IntegerType()))
    ])))

    # Apply UDF to create bands column
    bands_df = user_signatures.withColumn("bands", split_udf(col("minhashSignature")))

    # Step 2: Explode the bands so we get one row per (user, band)
    exploded_bands = bands_df.select(
        col("userId"),
        explode(col("bands")).alias("band_info")
    ).select(
        col("userId"),
        col("band_info.band_idx").alias("band_idx"),
        col("band_info.band").alias("band")
    )

    # Step 3: Group users by (band_idx, band)
    grouped_bands = exploded_bands.groupBy("band_idx", "band").agg(
        collect_list("userId").alias("candidate_users")
    )

    return grouped_bands



# --- Step 5: Find Top 100 Most Similar Pairs ---
def find_top_100_pairs(grouped_bands):
    # Step 1: Filter bands with more than 1 user (collisions only)
    filtered_bands = grouped_bands.filter(size(col("candidate_users")) > 1)

    # Step 2: Create all unique user pairs from candidate_users
    def create_user_pairs(users):
        users = sorted(users)
        pairs = []
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                pairs.append((users[i], users[j]))
        return pairs

    create_pairs_udf = udf(create_user_pairs, ArrayType(StructType([
        StructField("user1", IntegerType()),
        StructField("user2", IntegerType())
    ])))

    # Step 3: Explode into (user1, user2) rows
    exploded_pairs = filtered_bands.withColumn("user_pairs", explode(create_pairs_udf(col("candidate_users")))) \
                                   .select(
                                       col("user_pairs.user1").alias("user1"),
                                       col("user_pairs.user2").alias("user2")
                                   )

    # Step 4: Group by user pairs and count number of collisions
    pair_counts = exploded_pairs.groupBy("user1", "user2").count()

    # Step 5: Order by count descending and take top 100
    top_100_pairs = pair_counts.orderBy(col("count").desc()).limit(100)

    return top_100_pairs    



from pyspark.sql import Row
from pyspark.sql.functions import col, size, explode

def main():
    # --- Paths ---
    parquet_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.parquet"
    csv_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.csv"

    # --- Start Spark ---
    spark = start_spark()

    # --- Load Ratings ---
    ratings = load_ratings(spark, parquet_path, csv_path)

    # --- Build User Movie Sets ---
    user_movies_full = build_user_movie_sets(ratings)

    # --- Diagnostics BEFORE filtering ---
    print("=== Full Dataset Diagnostics ===")
    user_signatures_full = generate_minhash_signatures(user_movies_full)
    print("Number of users (full):", user_signatures_full.count())
    print("Unique MinHash signatures (full):", user_signatures_full.select("minhashSignature").distinct().count())
    user_signatures_full.groupBy("minhashSignature").count().orderBy(col("count").desc()).show(10)
    user_signatures_full.select(size("movieSet").alias("numMovies")).summary().show()

    # --- FILTER users with <5 movies ---
    user_movies = user_movies_full.filter(size(col("movieSet")) >= 3)
    print("Number of users after filtering (movieSet size ≥ 3):", user_movies.count())

    # --- Generate filtered MinHash signatures ---
    user_signatures = generate_minhash_signatures(user_movies)

    # --- Diagnostics AFTER filtering ---
    print("Unique MinHash signatures (filtered):", user_signatures.select("minhashSignature").distinct().count())
    user_signatures.groupBy("minhashSignature").count().orderBy(col("count").desc()).show(10)
    user_signatures.select(size("movieSet").alias("numMovies")).summary().show()
    user_signatures.select("userId", "movieSet", "minhashSignature").show(5, truncate=False)

    # --- Cluster Inspection: Top MinHash Signature ---
    top_sig = user_signatures.groupBy("minhashSignature") \
        .count() \
        .orderBy(col("count").desc()) \
        .limit(1) \
        .collect()[0]["minhashSignature"]

    sig_df = spark.createDataFrame([Row(minhashSignature=top_sig)])
    matching_users = user_signatures.join(sig_df, on="minhashSignature", how="inner")

  #  print("=== Users with Most Frequent MinHash Signature ===")
 #   matching_users.select("userId", "movieSet").show(20, truncate=False)

    # Explode movie sets into individual movieIds
    exploded_movies = matching_users.select(explode(col("movieSet")).alias("movieId"))

  #  print("=== Unique Movies Rated by This Cluster ===")
  #  exploded_movies.distinct().orderBy("movieId").show(100, truncate=False)

 #   print("=== Most Frequently Rated Movies in the Cluster ===")
  #  exploded_movies.groupBy("movieId").count().orderBy(col("count").desc()).show(20)

    # --- Apply LSH ---
    grouped_bands = apply_lsh(user_signatures)
    print("Number of band buckets:", grouped_bands.count())
    grouped_bands_with_size = grouped_bands.withColumn("group_size", size(col("candidate_users")))
    grouped_bands_with_size.select("group_size").summary().show()
    print("Number of buckets with >1 user:", grouped_bands_with_size.filter(col("group_size") > 1).count())
    grouped_bands_with_size.selectExpr("max(group_size)").show()

    # --- Find Top 100 pairs ---
    top_100_pairs = find_top_100_pairs(grouped_bands)
   # print("Top 100 user pairs by band collision count:")
   # top_100_pairs.show(100, truncate=False)

    # --- Save result ---
    print("saving to parquet")
    top_100_pairs.write.parquet("top_100_pairs_large-final.parquet", mode="overwrite")

    # --- Stop Spark ---
    spark.stop()


if __name__ == "__main__":
    main()
