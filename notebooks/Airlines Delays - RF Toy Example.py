# Databricks notebook source
# MAGIC %md #### Create toy example of random forests for final notebook

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
from pyspark.ml.classification import RandomForestClassifier as RF, DecisionTreeClassifier as DT
from pyspark.ml import Pipeline

# COMMAND ----------

# initialize the sql context
sqlContext = SQLContext(sc)

# COMMAND ----------

# global variables

# shared directory for our team (make sure it exists)
final_project_path = "dbfs:/mnt/mids-w261/group_5/"
dbutils.fs.mkdirs(final_project_path)

# output paths
train_data_output_path = final_project_path + "training_data_output/train.parquet"
test_data_output_path = final_project_path + "training_data_output/test.parquet"
train_toy_output_path = final_project_path + "training_data_output/train_toy.parquet"
test_toy_output_path = final_project_path + "training_data_output/test_toy.parquet"

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/mnt/mids-w261/group_5"))

# COMMAND ----------

# Read in parquet file
train_set = spark.read.parquet(train_data_output_path)

# COMMAND ----------

# MAGIC %md ## Algorithm Implementation

# COMMAND ----------

# MAGIC %md We have selected Random Forests (RF) as the final model based on results from the above exploritory algorithm analysis. We will demonstrate a decision tree classifier using a toy example with 3 features from the flight delay training data and continue with an explaination of the RF algorithm. 

# COMMAND ----------

# MAGIC %md #### Toy Example

# COMMAND ----------

# MAGIC %md ##### Example Data
# MAGIC First we separate 20% of the training data that the toy model will use to make predictions. Then we select 3 features from the dataset to visualize the trees that RF will build and compile these into a feature vector for the model.

# COMMAND ----------

# Divide training data into train and test set with features for example
train_toy, test_toy = train_set.select("label", "fl_date", "PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "day_of_week_Index", "crs_dep_hour_Index", "month_Index").randomSplit([0.8, 0.2], seed = 1)

# COMMAND ----------

# Select 3 features and compile into feature vector
features = ["PREVIOUS_FLIGHT_DELAYED_FOR_MODELS_Index", "origin_avg_dep_delay", "crs_dep_hour_Index"]
assembler = VectorAssembler(inputCols=features, outputCol="features").setHandleInvalid("keep")

train_toy = assembler.transform(train_toy)
test_toy = assembler.transform(test_toy)

# COMMAND ----------

train_toy.write.format("parquet").mode("overwrite").save(train_toy_output_path)
test_toy.write.format("parquet").mode("overwrite").save(test_toy_output_path)

# COMMAND ----------

train_toy = spark.read.option("header", "true").parquet(train_toy_output_path)
test_toy = spark.read.option("header", "true").parquet(test_toy_output_path)

# COMMAND ----------

# MAGIC %md ##### Training Decision Trees   
# MAGIC 
# MAGIC Next we will train a decision tree. Each tree is constructed with a series of splitting rules. The example figure below builds one tree with 3 available features. The first node at the top of the tree is split based on whether the previous flight is delayed. From the right branch the next split is on scheduled departure hour of day. The tree increases depth by choosing the best split considering all features and split points. These splits divide the training examples into 7 regions, at the leaf nodes, based on the combination of their features. 
# MAGIC 
# MAGIC How does the model decide splits? Our classification tree splits at the point which minimizes the *gini index*, a measure of node purity. The equation for the gini index is shown below, where \\(\hat{p}\_{mk}\\) is the proportion of examples in region \\(m\\) of class \\(k\\). 
# MAGIC 
# MAGIC $$ G = \sum\_{k=1}^{K} {\hat{p}\_{mk} (1 - \hat{p}\_{mk})} $$
# MAGIC 
# MAGIC The gini index will be minimized when \\(\hat{p}\_{m, k=0}\\) and \\(\hat{p}\_{m, k=1}\\) are close to 0 or 1, or when almost all the flight examples in the region are either delayed or not delayed. 

# COMMAND ----------

# Simple decision tree model
dt = DT(labelCol="label", featuresCol="features")
DT_model = dt.fit(train_toy)

display(DT_model)

# COMMAND ----------

# MAGIC %md ##### Make Predictions
# MAGIC To make a prediction using the decision tree, we assign a test data point to the leaf node (region) of the tree to which it belongs based on its features. The predicted class for a test example in region \\(m\\) is \\(argmax\_k\\) \\(\hat{p}\_{mk}\\), or the majority class.  
# MAGIC 
# MAGIC Below is an example of a prediction on a test example. For this example, the previous flight for the aircraft was delayed (feature 0 = 1) which moves down the left branch from the top of the tree. This flight's departure time is in hour 15, which moves it down the right branch of the next node. Next, the average delay at the origin airport 3 hours before is 12.9 minutes, which is less than the split point at 14.6 minutes. Lastly, repeating the hour of day feature for a split increases node purity and this flight is predicted to have no delay.

# COMMAND ----------

# Add row number to compare predictions for the same test example
test_toy = test_toy.withColumn("row", f.monotonically_increasing_id())

# COMMAND ----------

# Predict on toy test set
pred_toy_DT = DT_model.transform(test_toy)

# Create dataframe with predictions and show example
labelAndPrediction = pred_toy_DT.select("label", "row", "prediction", "features")
display(labelAndPrediction.where(labelAndPrediction.row == 575525629176))

# COMMAND ----------

# MAGIC %md ##### RF Algorithm
# MAGIC The method of averaging many trees grown from repeated samples of the training data, or bagging, decreases variance of the model that would occur with any one tree. Bagging grows deep trees and does not prune. The RF training method goes a step further to help guarantee a more reliable result. RF trees are built such that each node is randomly assigned a subset of features that will be considered as possible split candidates. This means that the trees will differ from each other, which when averaged will decrease variance more than bagging alone.  
# MAGIC 
# MAGIC Below we will train a RF model on the same data using 3 trees.

# COMMAND ----------

# RF model
rf = RF(labelCol="label", featuresCol="features", numTrees=3, maxDepth=5)
RF_model = rf.fit(train_toy)

# COMMAND ----------

# Print tree nodes for all RF trees
print(RF_model.toDebugString)

# COMMAND ----------

# MAGIC %md ##### Prediction with RF
# MAGIC RF then combines these predictions for all trees using a majority vote. If \\(\hat{p}\_{n,k}\\) is the proportion of predictions for class \\(k\\) over \\(n\\) trees, the majority vote is \\(argmax\_k\\) \\(\hat{p}\_{n,k}\\).

# COMMAND ----------

# Predict on toy test set with RF
pred_toy_RF = RF_model.transform(test_toy)

# Create dataframe with predictions and show example
labelAndPrediction_RF = pred_toy_RF.select("label", "row", "prediction", "features")
display(labelAndPrediction_RF.where(labelAndPrediction_RF.row == 575525629176))

# COMMAND ----------

# MAGIC %md The above test example has the following features: previous flight delayed, average delay at origin airport = 12.9 minutes, scheduled hour of departure delay = 15. The RF model predicts this as a delay.

# COMMAND ----------

# MAGIC %md #### Find test example to use (not for final notebook)

# COMMAND ----------

labelAndPrediction_RF_join = labelAndPrediction_RF.withColumnRenamed("prediction", "prediction_RF").select("prediction_RF", "row")
labelAndPrediction_join = labelAndPrediction_RF_join.join(labelAndPrediction, "row", "inner")
display(labelAndPrediction_join.sample(False, 0.0001))

# COMMAND ----------

display(labelAndPrediction_join.where((f.col("label") == f.col("prediction_RF")) & (f.col("label") != f.col("prediction"))))