from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min, max

def start_spark():
    return SparkSession.builder.appName("Validate_Splits").getOrCreate()

def load_split(spark, path):
    return spark.read.parquet(path)

def main():
    spark = start_spark()

    base_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits"
    train = load_split(spark, f"{base_path}/train_ratings.parquet")
    val = load_split(spark, f"{base_path}/val_ratings.parquet")
    test = load_split(spark, f"{base_path}/test_ratings.parquet")

    # Show sizes
    print(f"Train size: {train.count()}", flush=True)
    print(f"Validation size: {val.count()}", flush=True)
    print(f"Test size: {test.count()}", flush=True)

    # Check for overlap
    train_ids = train.select("userId", "movieId")
    val_ids = val.select("userId", "movieId")
    test_ids = test.select("userId", "movieId")

    print("Checking overlaps...", flush=True)
    print("Train ∩ Val:", train_ids.intersect(val_ids).count(), flush=True)
    print("Train ∩ Test:", train_ids.intersect(test_ids).count(), flush=True)
    print("Val ∩ Test:", val_ids.intersect(test_ids).count(), flush=True)

    # Verify temporal order
    print("Timestamp ranges:", flush=True)
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        min_ts = df.agg(min("timestamp")).first()[0]
        max_ts = df.agg(max("timestamp")).first()[0]
        print(f"{name} → Min: {min_ts}, Max: {max_ts}", flush=True)

    spark.stop()

if __name__ == "__main__":
    main()

