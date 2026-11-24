from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MaxTempRDD").getOrCreate()
rdd = spark.sparkContext.textFile("temps.txt")

max_temp = (rdd.map(lambda line: line.split(","))
               .filter(lambda f: len(f) == 2)
               .map(lambda f: (f[0].strip('" '), float(f[1])))
               .reduceByKey(max)                      # max per city
               .map(lambda kv: (kv[1], kv[0]))
               .max())                                # overall max

print("Maximum temperature:", max_temp[0], "in", max_temp[1])
spark.stop()
