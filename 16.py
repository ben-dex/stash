from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder.appName("Exercise16_Multiclass_Image_NB").getOrCreate()

# 2. Create dummy multi-class image dataset (4 pixel intensities across 3 classes: 0.0, 1.0, 2.0)
dummy_multiclass_images = [
    (0.90, 0.85, 0.10, 0.15, 0.0),
    (0.88, 0.82, 0.05, 0.12, 0.0),
    (0.92, 0.80, 0.12, 0.08, 0.0),
    (0.10, 0.15, 0.90, 0.85, 1.0),
    (0.12, 0.08, 0.88, 0.91, 1.0),
    (0.05, 0.10, 0.86, 0.89, 1.0),
    (0.85, 0.10, 0.12, 0.90, 2.0),
    (0.82, 0.15, 0.08, 0.88, 2.0),
    (0.87, 0.09, 0.10, 0.92, 2.0),
]
cols = ["pixel1", "pixel2", "pixel3", "pixel4", "label"]
df = spark.createDataFrame(dummy_multiclass_images, cols)

# 3. Assemble pixel columns into a single features vector
pixel_cols = ["pixel1", "pixel2", "pixel3", "pixel4"]
assembler = VectorAssembler(inputCols=pixel_cols, outputCol="features")
feature_df = assembler.transform(df)

# 4. Train built-in NaiveBayes model for multi-class classification
nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    modelType="gaussian",
)
model = nb.fit(feature_df)

# 5. Predict classes, inspect posterior probabilities, and compute accuracy
predictions = model.transform(feature_df)
predictions.select(
    "features", "label", "prediction", "probability"
).show(truncate=False)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Multi-class Accuracy Score: {accuracy:.4f}")

spark.stop()