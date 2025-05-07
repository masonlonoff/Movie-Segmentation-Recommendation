from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import collect_list, col, lit


def start_spark():
    return SparkSession.builder.appName("ImplicitALSModelEval").getOrCreate()


def load_and_prepare_data(spark, train_path, val_path, test_path):
    def to_implicit(df):
        return df.select("userId", "movieId").withColumn("rating", lit(1.0))

    train_df = to_implicit(spark.read.parquet(train_path).cache())
    val_df = to_implicit(spark.read.parquet(val_path).cache())
    test_df = to_implicit(spark.read.parquet(test_path).cache())
    return train_df, val_df, test_df


def train_als_model(train_df, val_df, ranks, reg_params):
    best_model = None
    best_ndcg = -1
    best_rank = None
    best_reg = None

    for rank in ranks:
        for reg in reg_params:
            print(f"Training ALS (implicit) with rank={rank}, regParam={reg}", flush=True)
            als = ALS(
                userCol="userId",
                itemCol="movieId",
                ratingCol="rating",
                rank=rank,
                regParam=reg,
                implicitPrefs=True,
                coldStartStrategy="drop",
                nonnegative=True
            )

            model = als.fit(train_df)

            preds = model.recommendForAllUsers(100).select("userId", "recommendations.movieId")
            actuals = val_df.groupBy("userId").agg(collect_list("movieId").alias("actual"))
            joined = preds.join(actuals, on="userId").rdd.map(lambda row: (row["movieId"], row["actual"]))

            metrics = RankingMetrics(joined)
            ndcg = metrics.ndcgAt(100)
            print(f"Rank={rank}, RegParam={reg}, NDCG@100={ndcg:.4f}", flush=True)

            if ndcg > best_ndcg:
                best_model = model
                best_ndcg = ndcg
                best_rank = rank
                best_reg = reg

    print(f"Best Implicit ALS - Rank: {best_rank}, RegParam: {best_reg}, NDCG@100: {best_ndcg:.4f}", flush=True)
    return best_model


def evaluate_model(model, test_df):
    preds = model.recommendForAllUsers(100).select("userId", "recommendations.movieId")
    actuals = test_df.groupBy("userId").agg(collect_list("movieId").alias("actual"))
    joined = preds.join(actuals, on="userId").rdd.map(lambda row: (row["movieId"], row["actual"]))

    metrics = RankingMetrics(joined)
    print(f"Precision@100: {metrics.precisionAt(100):.4f}", flush=True)
    print(f"Recall@100: {metrics.recallAt(100):.4f}", flush=True)
    print(f"NDCG@100: {metrics.ndcgAt(100):.4f}", flush=True)
    print(f"MAP@100: {metrics.meanAveragePrecisionAt(100):.4f}", flush=True)


def main():
    spark = start_spark()

    train_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits/train_ratings.parquet"
    val_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits/val_ratings.parquet"
    test_path = "hdfs:///user/ml9542_nyu_edu/ml-latest/splits/test_ratings.parquet"

    train_df, val_df, test_df = load_and_prepare_data(spark, train_path, val_path, test_path)

    ranks = [18,22, 25]
    reg_params = [0.125, 0.15, 0.2]
    best_model = train_als_model(train_df, val_df, ranks, reg_params)

    evaluate_model(best_model, test_df)
    spark.stop()


if __name__ == "__main__":
    main()

