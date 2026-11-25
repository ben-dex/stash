from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder.appName("Exercise11_Clustering").getOrCreate()

# 2. Create dummy dataset (e.g., 2D feature points forming roughly 2 clusters)
dummy_data = [
    (1.0, 1.2),
    (1.4, 0.8),
    (0.8, 1.5),
    (1.2, 1.1),
    (8.0, 8.2),
    (8.5, 7.9),
    (7.8, 8.6),
    (8.2, 8.1),
]
columns = ["feature1", "feature2"]
df = spark.createDataFrame(dummy_data, columns)

# 3. Assemble features into a single vector column (required by Spark ML)
assembler = VectorAssembler(inputCols=columns, outputCol="features")
feature_df = assembler.transform(df)

# 4. Train built-in KMeans clustering model
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=2, seed=42)
model = kmeans.fit(feature_df)

# 5. Predict clusters and inspect results
predictions = model.transform(feature_df)
predictions.select("feature1", "feature2", "cluster").show()

# 6. Evaluate clustering using Silhouette score
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette:.4f}")

# Print cluster centers
print("Cluster Centers:")
for idx, center in enumerate(model.clusterCenters()):
    print(f"  Cluster {idx}: {center}")

spark.stop()
