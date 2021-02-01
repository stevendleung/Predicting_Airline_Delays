# Databricks notebook source
# MAGIC %md
# MAGIC # GBT Model

# COMMAND ----------

# package imports
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

# model imports
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# COMMAND ----------

# initialize the sql context
sqlContext = SQLContext(sc)

# COMMAND ----------

# global variables

# shared directory for our team (make sure it exists)
final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

# Processed data files for exploratory models
train_data_exploratory_path = final_project_path + "training_data_output/train_exploratory.parquet"
val_data_exploratory_path = final_project_path + "training_data_output/val_exploratory.parquet"
test_data_exploratory_path = final_project_path + "training_data_output/test_exploratory.parquet"

# COMMAND ----------

# Read in parquet file
train_GBT = spark.read.parquet(train_data_exploratory_path)
val_GBT = spark.read.parquet(val_data_exploratory_path)
test_GBT = spark.read.parquet(test_data_exploratory_path)

# COMMAND ----------

# Assemble categorical and numeric features into vector
features = ["month_Index", "day_of_week_Index", "op_unique_carrier_Index", "Holiday_Index", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_WND_type_code_Index", "origin_CIG_ceiling_visibility_okay_Index", "origin_VIS_variability_Index", "dest_WND_type_code_Index", "dest_CIG_ceiling_visibility_okay_Index", "dest_VIS_variability_Index", "crs_dep_hour_Index", "origin_num_flights","origin_avg_dep_delay", "origin_pct_dep_del15", "origin_avg_taxi_time", "origin_avg_weather_delay", "origin_avg_nas_delay", "origin_avg_security_delay", "origin_avg_late_aircraft_delay", "dest_num_flights","dest_avg_dep_delay", "dest_pct_dep_del15", "dest_avg_taxi_time", "dest_avg_weather_delay", "dest_avg_nas_delay", "dest_avg_security_delay", "dest_avg_late_aircraft_delay", "carrier_num_flights", "carrier_avg_dep_delay", "carrier_avg_carrier_delay", "origin_WND_direction_angle", "origin_WND_speed_rate", "origin_CIG_ceiling_height", "origin_VIS_distance", "origin_TMP_air_temperature", "origin_SLP_sea_level_pressure", "dest_WND_direction_angle", "dest_WND_speed_rate", "dest_CIG_ceiling_height", "dest_VIS_distance", "dest_TMP_air_temperature", "dest_SLP_sea_level_pressure"]

assembler = VectorAssembler(inputCols=features, outputCol="features").setHandleInvalid("skip")

train_GBT = assembler.transform(train_GBT)
val_GBT = assembler.transform(val_GBT)
test_GBT = assembler.transform(test_GBT)

# COMMAND ----------

# MAGIC %md #### Train without cross validation

# COMMAND ----------

# Define GBT model
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter = 100, maxDepth=10)

# Train GBT model with cross validation
GBT_model = gbt.fit(train_GBT)

# COMMAND ----------

def make_predictions(model, testSet):
    '''Predict on validation/test set and print accuracy, recall, precision, and f1-score.'''
    predictions = model.transform(testSet)

    labelAndPrediction = predictions.select("label", "prediction", "features")
    
    # evaluate model on validation set
    TP = labelAndPrediction.where((labelAndPrediction.label == 1) & (labelAndPrediction.prediction == 1)).count()
    FP = labelAndPrediction.where((labelAndPrediction.label == 0) & (labelAndPrediction.prediction == 1)).count()
    TN = labelAndPrediction.where((labelAndPrediction.label == 0) & (labelAndPrediction.prediction == 0)).count()
    FN = labelAndPrediction.where((labelAndPrediction.label == 1) & (labelAndPrediction.prediction == 0)).count()
    accuracy = (TP + TN)/labelAndPrediction.count()

    if TP + FN == 0:
        recall = 0
        precision = float(TP) / (TP + FP)

    elif TP + FP == 0:
        recall = float(TP) / (TP + FN)
        precision = 0

    else:
        recall = float(TP) / (TP + FN)
        precision = float(TP) / (TP + FP)

    if precision + recall == 0:
        f1_score = 0

    else:
      f1_score = 2 * ((precision * recall)/(precision + recall))  

    print('Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1-score: {:.3f}'.format(f1_score))

# COMMAND ----------

# make predictions on validation set
make_predictions(GBT_model, val_GBT)

# COMMAND ----------

# get feature importances
for i in range(len(features)):
    print("{}: {}".format(features[i],round(GBT_model.featureImportances[i],3)))

# COMMAND ----------

# make predictions on test set
make_predictions(GBT_model, test_GBT)

# COMMAND ----------

# MAGIC %md #### Train with cross validation

# COMMAND ----------

train_cv = train_GBT.union(val_GBT)

# COMMAND ----------

# set parameter search grid
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [1, 10])\
  .build()

# options for classification evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label")

# Cross validation
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds = 3)

# Train GBT model with cross validation
cv_model = cv.fit(train_cv)

# COMMAND ----------

# get best model from cross validation
best_GBT = cv_model.bestModel

# COMMAND ----------

# extract parameters
best_GBT.extractParamMap()

# COMMAND ----------

# make predictions on test set
make_predictions(cv_model, test_GBT)

# COMMAND ----------

# get feature importances
for i in range(len(features)):
    print("{}: {}".format(features[i],round(cv_model.featureImportances[i],3)))