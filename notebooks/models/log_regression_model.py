# Databricks notebook source
# MAGIC %md 
# MAGIC ## Base Model - Logistic Regression

# COMMAND ----------

##Import packages
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType
from pyspark.sql import SQLContext
from pyspark.sql import types
from pyspark.sql.functions import col, lag, udf, to_timestamp, monotonically_increasing_id
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime, timedelta
from pyspark.ml.feature import IndexToString, StringIndexer, OneHotEncoder, VectorAssembler, Bucketizer, StandardScaler
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

import mlflow
import mlflow.spark

# COMMAND ----------

#Set Path for one hot encoded files
final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

train_data_output_path_one_hot = final_project_path + "training_data_output/train_one_hot.parquet"
test_data_output_path_one_hot = final_project_path + "training_data_output/test_one_hot.parquet"

# Read in parquet file
train_log = spark.read.parquet(train_data_output_path_one_hot)
test_log = spark.read.parquet(test_data_output_path_one_hot)

print(train_log.count())
print(test_log.count())

# COMMAND ----------

#Select model features
selected_features  = ["month_Indicator","day_of_week_Indicator","crs_dep_hour_Indicator","op_unique_carrier_Indicator","origin_WND_speed_rate","origin_CIG_ceiling_height","origin_VIS_distance","origin_TMP_air_temperature","Holiday_Indicator","PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Indicator","label"]

#Drop nulls for the training and test set 
training_set = train_log.select(selected_features).dropna()
test_set = test_log.select(selected_features).dropna()

#Convert outcome variable to integer type 
training_set = training_set.withColumn('label', training_set['label'].cast(IntegerType()))  
test_set = test_set.withColumn('label', test_set['label'].cast(IntegerType()))  

print(training_set.count())
print(test_set.count())

# COMMAND ----------

def metrics(df):
  
#Calculate metrics to evaluate accuracy, precsision, recall and f1 score for model
  true_positive = df[(df.label == 1) & (df.prediction == 1)].count()
  true_negative = df[(df.label == 0) & (df.prediction == 0)].count()
  false_positive = df[(df.label == 0) & (df.prediction == 1)].count()
  false_negative = df[(df.label == 1) & (df.prediction == 0)].count()
  accuracy = ((true_positive + true_negative)/df.count())

  if(true_positive + false_negative == 0.0):
    recall = 0.0
    precision = float(true_positive) / (true_positive + false_positive)
    
  elif(true_positive + false_positive == 0.0):
    recall = float(true_positive) / (true_positive + false_negative)
    precision = 0.0
    
  else:
    recall = float(true_positive) / (true_positive + false_negative)
    precision = float(true_positive) / (true_positive + false_positive)

  if(precision + recall == 0):
    f1_score = 0
    
  else:
    f1_score = 2 * ((precision * recall)/(precision + recall))   
    
  print("Accuracy:", accuracy)
  print("Recall:", recall)
  print("Precision: ", precision)
  print("F1 score:", f1_score)  

# COMMAND ----------

train_cols = training_set.columns
train_cols.remove("label")

#Combine training input columns into a single vector
assembler = VectorAssembler(inputCols=train_cols,outputCol="features").setHandleInvalid("keep")

#Scale features 
standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")

lr = LogisticRegression(labelCol="label", featuresCol="Scaled_features")
pipeline = Pipeline(stages=[assembler,standardscaler,lr])

paramGrid = ParamGridBuilder() \
    .addGrid(lr.threshold, [.185]) \
    .addGrid(lr.maxIter, [10,20]) \
    .addGrid(lr.regParam, [0.1]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=3) 

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

cvModel = crossval.fit(training_set)

# COMMAND ----------

#Review best model parameters
best_model = cvModel.bestModel
parameter_dict = best_model.stages[-1].extractParamMap()

parameter_dict_blank = {}
for x, y in parameter_dict.items():
  parameter_dict_blank[x.name] = y

print("Best Regularization Parameter",parameter_dict_blank["regParam"])
print("Best Iteration Parameter",parameter_dict_blank["maxIter"])
print("Best Threshold Parameter",parameter_dict_blank["threshold"])

# COMMAND ----------

#Review performance on training data 
train_model = cvModel.transform(training_set)
train_metric = evaluator.evaluate(train_model)
print("Train_Area_UnderPR",train_metric)
print("")
print("Train_Performance")
metrics(train_model)

# COMMAND ----------

#Review performance on test data 
test_model = cvModel.transform(test_set)
test_metric = evaluator.evaluate(test_model)
print("Test_Area_UnderPR",test_metric)
print("")
print("Test_Performance")
metrics(test_model)