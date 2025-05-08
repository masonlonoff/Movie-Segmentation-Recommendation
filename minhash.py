from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from pyspark.sql.functions import collect_set, collect_list, col, explode, udf, array_intersect, size, array_intersect, array_union
import numpy as np
from pyspark.sql import Row


# Initialize Spark
def start_spark():
    spark = SparkSession.builder.appName("Capstone_MinHash").getOrCreate()
    return spark


# Load in the data
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

# Build the user-movie sets
def build_user_movie_sets(ratings):
    user_movies = ratings.groupBy("userId").agg(collect_set("movieId").alias("movieSet"))
    return user_movies

# Creating min hash signatures
def generate_minhash_signatures(user_movies, num_hashes=128):
    np.random.seed(11)

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

# Applying LSH
def apply_lsh(user_signatures, num_bands=32, rows_per_band=4):
    
    def split_into_bands(signature):
        bands = []
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band = signature[start:end]
            bands.append((band_idx, tuple(band)))
        return bands

    
    split_udf = udf(split_into_bands, ArrayType(StructType([
        StructField("band_idx", IntegerType()),
        StructField("band", ArrayType(IntegerType()))
    ])))

    
    bands_df = user_signatures.withColumn("bands", split_udf(col("minhashSignature")))

    
    exploded_bands = bands_df.select(
        col("userId"),
        explode(col("bands")).alias("band_info")
    ).select(
        col("userId"),
        col("band_info.band_idx").alias("band_idx"),
        col("band_info.band").alias("band")
    )

    grouped_bands = exploded_bands.groupBy("band_idx", "band").agg(
        collect_list("userId").alias("candidate_users")
    )

    return grouped_bands



# Finding top pairs
def find_top_100_pairs(grouped_bands):
    
    filtered_bands = grouped_bands.filter(size(col("candidate_users")) > 1)

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


    exploded_pairs = filtered_bands.withColumn("user_pairs", explode(create_pairs_udf(col("candidate_users")))) \
                                   .select(
                                       col("user_pairs.user1").alias("user1"),
                                       col("user_pairs.user2").alias("user2")
                                   )

    
    pair_counts = exploded_pairs.groupBy("user1", "user2").count()

    top_100_pairs = pair_counts.orderBy(col("count").desc()).limit(100)

    return top_100_pairs    



def main():
    # Paths
    parquet_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.parquet"
    csv_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/ratings.csv"

    # starting spark
    spark = start_spark()

    # loading ratings
    ratings = load_ratings(spark, parquet_path, csv_path)

    # building the user sets
    user_movies_full = build_user_movie_sets(ratings)

    # Pre filtering diagnostics
    print("=== Full Dataset Diagnostics ===")
    user_signatures_full = generate_minhash_signatures(user_movies_full)
    print("Number of users (full):", user_signatures_full.count())
    print("Unique MinHash signatures (full):", user_signatures_full.select("minhashSignature").distinct().count())
    user_signatures_full.groupBy("minhashSignature").count().orderBy(col("count").desc()).show(10)
    user_signatures_full.select(size("movieSet").alias("numMovies")).summary().show()

    # Filter out movies with < 3 users
    user_movies = user_movies_full.filter(size(col("movieSet")) >= 3)
    print("Number of users after filtering (movieSet size ≥ 3):", user_movies.count())

    # Generating the minhash signatures
    user_signatures = generate_minhash_signatures(user_movies)

    # Post filtering diagnostics
    print("Unique MinHash signatures (filtered):", user_signatures.select("minhashSignature").distinct().count())
    user_signatures.groupBy("minhashSignature").count().orderBy(col("count").desc()).show(10)
    user_signatures.select(size("movieSet").alias("numMovies")).summary().show()
    user_signatures.select("userId", "movieSet", "minhashSignature").show(5, truncate=False)


    # Examining top clusters
    top_sig = user_signatures.groupBy("minhashSignature") \
        .count() \
        .orderBy(col("count").desc()) \
        .limit(1) \
        .collect()[0]["minhashSignature"]

    sig_df = spark.createDataFrame([Row(minhashSignature=top_sig)])
    matching_users = user_signatures.join(sig_df, on="minhashSignature", how="inner")

    exploded_movies = matching_users.select(explode(col("movieSet")).alias("movieId"))


    # Applying LSH
    grouped_bands = apply_lsh(user_signatures)
    print("Number of band buckets:", grouped_bands.count())
    grouped_bands_with_size = grouped_bands.withColumn("group_size", size(col("candidate_users")))
    grouped_bands_with_size.select("group_size").summary().show()
    print("Number of buckets with >1 user:", grouped_bands_with_size.filter(col("group_size") > 1).count())
    grouped_bands_with_size.selectExpr("max(group_size)").show()

    # Finding top 100 
    top_100_pairs = find_top_100_pairs(grouped_bands)

    # adds movies back together for similarity
    user1_movies = user_movies.selectExpr("userId as user1", "movieSet as movieSet1")
    user2_movies = user_movies.selectExpr("userId as user2", "movieSet as movieSet2")

    # Join movie sets to top pairs
    top_100_pairs_with_sets = top_100_pairs \
        .join(user1_movies, on="user1") \
        .join(user2_movies, on="user2")

    # Compute Jaccard similarity
    top_100_final = top_100_pairs_with_sets.withColumn(
        "jaccard_similarity",
        size(array_intersect("movieSet1", "movieSet2")) / size(array_union("movieSet1", "movieSet2"))
)
    top_100_final.write.parquet("top_100_pairs_Jaccard.parquet", mode = "overwrite")

    # Save result
    print("saving to parquet")
    top_100_pairs.write.parquet("top_100_pairs_large-final.parquet", mode="overwrite")

    spark.stop()


if __name__ == "__main__":
    main()
