from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# 1. Initialize SparkSession
spark = SparkSession.builder.appName("Exercise21_KNN_Handwritten").getOrCreate()

# 2. Create dummy handwritten image dataset (4 pixel intensity features + digit label)
# e.g., digit 0.0 vs digit 1.0
dummy_data = [
    (0, 0.90, 0.85, 0.10, 0.05, 0.0),
    (1, 0.88, 0.82, 0.12, 0.08, 0.0),
    (2, 0.85, 0.80, 0.08, 0.10, 0.0),
    (3, 0.10, 0.12, 0.90, 0.88, 1.0),
    (4, 0.08, 0.15, 0.85, 0.92, 1.0),
    (5, 0.12, 0.10, 0.88, 0.86, 1.0),
]
cols = ["id", "p1", "p2", "p3", "p4", "label"]
df = spark.createDataFrame(dummy_data, cols)

k = 3
pixel_cols = ["p1", "p2", "p3", "p4"]

# 3. Compute pairwise Euclidean distance between test points (a) and training points (b)
a = df.alias("a")
b = df.alias("b")

dist_expr = F.sqrt(
    sum(F.pow(F.col(f"a.{c}") - F.col(f"b.{c}"), 2) for c in pixel_cols)
)

pairs = (
    a.crossJoin(b)
    .filter(F.col("a.id") != F.col("b.id"))  # leave-one-out neighbor pool
    .withColumn("distance", dist_expr)
)

# 4. Select top-k nearest neighbors per sample using built-in Window + row_number
w = Window.partitionBy("a.id").orderBy(F.col("distance").asc())
top_k = pairs.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= k)

# 5. Majority vote among top-k neighbors (ties broken by closest neighbor distance)
vote_window = Window.partitionBy("a.id").orderBy(
    F.col("votes").desc(), F.col("min_dist").asc()
)
votes = (
    top_k.groupBy("a.id", "a.label", "b.label")
    .agg(
        F.count("*").alias("votes"),
        F.min("distance").alias("min_dist"),
    )
    .withColumn("vote_rank", F.row_number().over(vote_window))
    .filter(F.col("vote_rank") == 1)
    .select(
        F.col("a.id").alias("id"),
        F.col("a.label").cast("double").alias("label"),
        F.col("b.label").cast("double").alias("prediction"),
    )
)

votes.orderBy("id").show()

# 6. Evaluate Accuracy using built-in MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(votes)
print(f"k-NN (k={k}) Accuracy Score: {accuracy:.4f}")

spark.stop()