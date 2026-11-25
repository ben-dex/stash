from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Binarizer, VectorAssembler
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder.appName(
    "Exercise15_Gaussian_vs_Bernoulli_NB"
).getOrCreate()

# 2. Create dummy image dataset (e.g., 4 pixel values representing continuous grayscale image intensity)
# Class 0.0: Digits with high top-left values | Class 1.0: Digits with high bottom-right values
dummy_image_data = [
    (0.9, 0.8, 0.1, 0.2, 0.0),
    (0.85, 0.95, 0.05, 0.1, 0.0),
    (0.92, 0.78, 0.15, 0.05, 0.0),
    (0.1, 0.15, 0.88, 0.92, 1.0),
    (0.2, 0.05, 0.95, 0.85, 1.0),
    (0.05, 0.12, 0.82, 0.9, 1.0),
]
cols = ["pixel1", "pixel2", "pixel3", "pixel4", "label"]
df = spark.createDataFrame(dummy_image_data, cols)

# 3. Assemble raw continuous pixel features (for Gaussian NB)
pixel_cols = ["pixel1", "pixel2", "pixel3", "pixel4"]
assembler = VectorAssembler(inputCols=pixel_cols, outputCol="gaussian_features")
continuous_df = assembler.transform(df)

# 4. Binarize pixels (0 or 1 thresholding) for Bernoulli NB (which expects binary/boolean features)
binarizer = Binarizer(
    threshold=0.5, inputCol="gaussian_features", outputCol="bernoulli_features"
)
binary_df = binarizer.transform(continuous_df)

# 5. Train & Evaluate Gaussian Naive Bayes (Continuous Features)
gnb = NaiveBayes(
    featuresCol="gaussian_features",
    labelCol="label",
    predictionCol="pred_gaussian",
    modelType="gaussian",
)
gnb_model = gnb.fit(binary_df)
pred_gnb = gnb_model.transform(binary_df)

evaluator_gnb = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="pred_gaussian", metricName="accuracy"
)
acc_gnb = evaluator_gnb.evaluate(pred_gnb)

# 6. Train & Evaluate Bernoulli Naive Bayes (Binary Features)
# 6. Train & Evaluate Bernoulli Naive Bayes (Binary Features)
bnb = NaiveBayes(
    featuresCol="bernoulli_features",
    labelCol="label",
    predictionCol="pred_bernoulli",
    rawPredictionCol="rawPred_bernoulli",
    probabilityCol="prob_bernoulli",
    modelType="bernoulli",
)
bnb_model = bnb.fit(binary_df)
pred_bnb = bnb_model.transform(pred_gnb)

evaluator_bnb = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="pred_bernoulli", metricName="accuracy"
)
acc_bnb = evaluator_bnb.evaluate(pred_bnb)

# 7. Compare Predictions and Accuracy Scores
pred_bnb.select(
    "gaussian_features",
    "bernoulli_features",
    "label",
    "pred_gaussian",
    "pred_bernoulli",
).show(truncate=False)

print(f"Gaussian Naive Bayes Accuracy:  {acc_gnb:.4f}")
print(f"Bernoulli Naive Bayes Accuracy: {acc_bnb:.4f}")
bnb_model = bnb.fit(binary_df)
pred_bnb = bnb_model.transform(pred_gnb)

evaluator_bnb = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="pred_bernoulli", metricName="accuracy"
)
acc_bnb = evaluator_bnb.evaluate(pred_bnb)

print(f"Gaussian Naive Bayes Accuracy:  {acc_gnb:.4f}")
print(f"Bernoulli Naive Bayes Accuracy: {acc_bnb:.4f}")

spark.stop()
