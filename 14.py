from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, RegexTokenizer
from pyspark.sql import SparkSession

# 1. Initialize SparkSession
spark = SparkSession.builder.appName("Exercise14_SMS_Spam_NaiveBayes").getOrCreate()

# 2. Create dummy SMS dataset (0.0 = Ham / Legitimate, 1.0 = Spam)
dummy_sms = [
    ("Hey are we still meeting for lunch today", 0.0),
    ("Can you send me the notes from class", 0.0),
    ("Let us catch up tomorrow morning", 0.0),
    ("Are you free for a quick call later", 0.0),
    ("WIN a free cash prize now click link", 1.0),
    ("Urgent claim your free lottery reward today", 1.0),
    ("Exclusive free gift voucher hurry and claim", 1.0),
    ("Congratulations you won a free prize claim now", 1.0),
]
columns = ["text", "label"]
df = spark.createDataFrame(dummy_sms, columns)

# 3. Tokenize text into word lists using built-in RegexTokenizer
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+")
words_df = tokenizer.transform(df)

# 4. Convert tokens to term-frequency vectors (CountVectorizer) and scale with TF-IDF
cv = CountVectorizer(inputCol="words", outputCol="raw_features")
cv_model = cv.fit(words_df)
featurized_df = cv_model.transform(words_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
rescaled_df = idf_model.transform(featurized_df)

# 5. Train built-in Multinomial Naive Bayes classifier (default modelType="multinomial")
nb = NaiveBayes(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    modelType="multinomial",
)
model = nb.fit(rescaled_df)

# 6. Predict and compute accuracy
predictions = model.transform(rescaled_df)
predictions.select("text", "label", "prediction", "probability").show(truncate=False)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy Score: {accuracy:.4f}")

spark.stop()