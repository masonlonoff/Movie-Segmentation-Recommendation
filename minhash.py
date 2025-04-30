from pyspark.sql import SparkSession
import random   
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from pyspark.sql.functions import collect_set, collect_list, col, explode, udf, array_intersect, size
from pyspark.sql.utils import AnalysisException
import os
import random 
import numpy as np

random.seed(11)


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
def generate_minhash_signatures(user_movies, num_hashes = 128):

    def create_hash_functions(num_hashes, max_val=100000, prime=104729):
        hash_functions = []
        for _ in range(num_hashes):
            a = random.randint(1, max_val)
            b = random.randint(0, max_val)

            def hash_fn(x, a=a, b=b, prime=prime):
                return (a * x + b) % prime

            hash_functions.append(hash_fn)

        return hash_functions

    hash_funcs = create_hash_functions(num_hashes)

    def minhash_movie_set(movie_set):
        signature = []
        for h in hash_funcs:
            min_hash = min([h(m) for m in movie_set])
            signature.append(min_hash)
        return signature


    minhash_udf = udf(minhash_movie_set, ArrayType(IntegerType()))
    user_signatures = user_movies.withColumn("minhashSignature", minhash_udf("movieSet"))

    return user_signatures
   

# --- Step 4: Apply LSH Bucketing ---
def apply_lsh(user_signatures, num_bands=16, rows_per_band=8):
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



def main():
    # --- Paths ---
    parquet_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.parquet"
    csv_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.csv"
    
    # --- Start Spark ---
    spark = start_spark()

    # --- Load Data (optimized) ---
    ratings = load_ratings(spark, parquet_path, csv_path)
    
    # --- Build movie sets ---
    user_movies = build_user_movie_sets(ratings)
    
    # --- Generate MinHash signatures ---
    user_signatures = generate_minhash_signatures(user_movies)
    user_signatures.select("userId", "movieSet", "minhashSignature").show(5, truncate=False)
    
    # --- Apply LSH bucketing ---
    grouped_bands = apply_lsh(user_signatures)
    print("Number of bands created:", grouped_bands.count())

    grouped_bands.show(25, truncate=False)

    from pyspark.sql.functions import size

# Add a group_size column
    grouped_bands_with_size = grouped_bands.withColumn("group_size", size(col("candidate_users")))

# Show basic stats
    grouped_bands_with_size.select("group_size").summary().show()

# How many bands have more than 1 candidate user
    print("Number of bands with >1 candidate:", grouped_bands_with_size.filter(col("group_size") > 1).count())

# Maximum group size
    grouped_bands_with_size.selectExpr("max(group_size)").show()


     # --- Find Top 100 most similar user pairs ---
    top_100_pairs = find_top_100_pairs(grouped_bands)
    
    top_100_pairs.write.parquet("top_100_pairs_large-3.parquet", mode="overwrite")


    # --- Stop Spark ---
    spark.stop()

if __name__ == "__main__":
    main()
