from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder.appName("Exercise13_NaiveBayes_Iris").getOrCreate()

# 2. Create dummy binary Iris dataset (Sepal Length, Sepal Width, Petal Length, Petal Width, Binary Species Label)
# Label 0.0: Setosa | Label 1.0: Versicolor
dummy_iris = [
    (5.1, 3.5, 1.4, 0.2, 0.0),
    (4.9, 3.0, 1.4, 0.2, 0.0),
    (4.7, 3.2, 1.3, 0.2, 0.0),
    (4.6, 3.1, 1.5, 0.2, 0.0),
    (7.0, 3.2, 4.7, 1.4, 1.0),
    (6.4, 3.2, 4.5, 1.5, 1.0),
    (6.9, 3.1, 4.9, 1.5, 1.0),
    (5.5, 2.3, 4.0, 1.3, 1.0),
]
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
df = spark.createDataFrame(dummy_iris, columns)

# 3. Assemble numeric features into a single features vector
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
feature_df = assembler.transform(df)

# 4. Train built-in NaiveBayes model (Gaussian model type for continuous data)
nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    modelType="gaussian",
)
model = nb.fit(feature_df)

# 5. Generate predictions and evaluated accuracy
predictions = model.transform(feature_df)
predictions.select(
    "sepal_length", "petal_length", "label", "prediction", "probability"
).show(truncate=False)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy Score: {accuracy:.4f}")

spark.stop()
