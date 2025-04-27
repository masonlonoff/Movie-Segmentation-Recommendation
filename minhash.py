from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_set
import random   
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType



# --- Step 0: Initialize Spark ---
def start_spark():
    spark = SparkSession.builder.appName("Capstone_MinHash").getOrCreate()
    return spark

# --- Step 1: Load Ratings Data ---
def load_ratings(spark, path):
    ratings = spark.read.csv(path, header=True, inferSchema=True)
    ratings = ratings.select("userId", "movieId").dropna()
    return ratings

# --- Step 2: Build User Movie Sets ---
def build_user_movie_sets(ratings):
    user_movies = ratings.groupBy("userId").agg(collect_set("movieId").alias("movieSet"))
    return user_movies

# --- Step 3: Generate MinHash Signatures ---
def generate_minhash_signatures(user_movies, num_hashes = 128):

    def create_hash_functions(num_hashes, max_val = 100000, prime = 104729):
        hash_functions = []
        for _ in range(num_hashes):
            a = random.randint(1, max_val)
            b = random.randint(0, max_val)

            def hash_fn(x, a=a, b=b, prime = prime):
                return (a*x + b) % prime

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
def apply_lsh(signatures):
    pass

# --- Step 5: Find Top 100 Most Similar Pairs ---
def find_top_100_pairs(candidates):
    pass



def main():
    # Adjust this when it's time
    ratings_path = "hdfs:///user/ml9542_nyu_edu/ml-latest-small/ratings.csv"
    
    spark = start_spark()
    ratings = load_ratings(spark, ratings_path)
    user_movies = build_user_movie_sets(ratings)

    
    user_signatures = generate_minhash_signatures(user_movies)

    user_signatures.select("userId", "movieSet", "minhashSignature").show(5, truncate=False)
    spark.stop()

if __name__ == "__main__":
    main()
