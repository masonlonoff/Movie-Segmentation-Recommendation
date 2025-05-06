from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, collect_list
from pyspark.mllib.evaluation import RankingMetrics


def start_spark():
    return SparkSession.builder.appName("PopularityBaselineEval").getOrCreate()


def load_data(spark, train_path, test_path):
    train_df = spark.read.parquet(train_path)
    test_df = spark.read.parquet(test_path)
    return train_df, test_df


def get_top_movies(train_df, k=100):
    return(
            train_df.groupBy("movieId")
            .agg(count("userId").alias("popularity"))
            .orderBy(col("popularity").desc())
            .limit(k)
            .select("movieId")
            .rdd.flatMap(lambda row: [row["movieId"]])
            .collect()
        )
        
def get_user_predictions(test_df, top_movies):
    return test_df.select("userId").distinct().rdd.map(lambda row: (row["userId"], top_movies))

def get_actual_interactions(test_df):
    return (
        test_df.groupBy("userId")
        .agg(collect_list("movieId").alias("actual_movies"))
        .rdd.map(lambda row: (row["userId"], row["actual_movies"]))
    )


def evaluate(predictions, actuals, k=100):
    joined = predictions.join(actuals).map(lambda x: (x[1][0], x[1][1]))
    metrics = RankingMetrics(joined)

    print("Sample joined prediction/actual pairs:")
    for row in joined.take(5):
        print("Predicted:", row[0])
        print("Actual   :", row[1])
        print()

    print(f"Precision@{k}: {metrics.precisionAt(k):.4f}")
    print(f"Recall@{k}: {metrics.recallAt(k):.4f}")
    print(f"NDCG@{k}: {metrics.ndcgAt(k):.4f}")
    print(f"MAP@{k}: {metrics.meanAveragePrecisionAt(k):.4f}")


def main():
    spark = start_spark()

    train_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits/train_ratings.parquet"
    test_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits/test_ratings.parquet"

    train_df, test_df = load_data(spark, train_path, test_path)

    print("Building popularity model from training data...")
    top_movies = get_top_movies(train_df, k=100)

    print("Generating predictions for validation users...")
    predictions = get_user_predictions(test_df, top_movies)
    actuals = get_actual_interactions(test_df)

    print("Evaluating popularity baseline...")
    evaluate(predictions, actuals, k=100)


    spark.stop()


if __name__ == "__main__":
    main()

