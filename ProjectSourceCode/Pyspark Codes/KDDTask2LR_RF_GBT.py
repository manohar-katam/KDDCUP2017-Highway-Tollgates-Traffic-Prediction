# Databricks notebook source
task2Data = spark.read.option("header","true").option("inferSchema","true").csv("/FileStore/tables/preprocessed_training_data_task2.csv")

# COMMAND ----------

display(task2Data)

# COMMAND ----------

task2Data.printSchema()

# COMMAND ----------

cols = task2Data.columns

# COMMAND ----------

task2Data.createOrReplaceTempView("data")

# COMMAND ----------

df = spark.sql("select 	`tollgate_id`,	`time_window`, direction, volume,  lag1, lag2, lag3, lag4, lag5, lag6, lag7,	Friday,	Monday,	Saturday,	Sunday,	Thursday,	Tuesday,	Wednesday,	hour__0,	hour__1,	hour__2,	hour__3,	hour__4,	hour__5,	hour__6,	hour__7,	hour__8,	hour__9,	hour__10,	hour__11,	hour__12,	hour__13,	hour__14,	hour__15,	hour__16,	hour__17,	hour__18,	hour__19,	hour__20,	hour__21,	hour__22,	hour__23,	`0`, `20`, `40`,	holiday,	index,	pressure, sea_pressure,	wind_direction,	wind_speed,	temperature,	rel_humidity,	precipitation, `volume` as label from data")
display(df)

# COMMAND ----------

## converting string values to sparse vectors using one hot encoding
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

categorical_Columns = ["time_window"]
pipeline_stages = [] # pipeline stages 
for categorical_Col in categorical_Columns:
  string_Indexer = StringIndexer(inputCol=categorical_Col, outputCol=categorical_Col+"Index")
  onehot_encoder = OneHotEncoder(inputCol=categorical_Col+"Index", outputCol=categorical_Col+"classVec")
  pipeline_stages += [string_Indexer, onehot_encoder]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numeric_Cols = ["tollgate_id", "direction", "lag1", "lag2","lag3","lag4","lag5","lag6","lag7",  "Friday",	"Monday",	"Saturday",	"Sunday",	"Thursday",	"Tuesday",	"Wednesday",	"hour__0",	"hour__1",	"hour__2",	"hour__3",	"hour__4",	"hour__5",	"hour__6",	"hour__7",	"hour__8",	"hour__9",	"hour__10",	"hour__11",	"hour__12",	"hour__13",	"hour__14",	"hour__15",	"hour__16",	"hour__17",	"hour__18",	"hour__19",	"hour__20",	"hour__21",	"hour__22",	"hour__23",	"0",	"20",		"40",		"holiday",	"index",	"pressure",	"wind_direction",	"wind_speed",	"temperature",	"rel_humidity",	"precipitation"]
assembler_Inputs = map(lambda c: c + "classVec", categorical_Columns) + numeric_Cols
vector_assembler = VectorAssembler(inputCols=assembler_Inputs, outputCol="features")
pipeline_stages += [vector_assembler]

# COMMAND ----------

pipeline = Pipeline(stages=pipeline_stages)

pipeline_Model = pipeline.fit(df)
df = pipeline_Model.transform(df)

# Keep relevant columns
selected_cols = ["label", "features"] + cols
df = df.select(selected_cols)
display(df)

# COMMAND ----------

selected_cols

# COMMAND ----------

trainingData = df
print trainingData.count()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

linear_reg = LinearRegression()

param_Grid = ParamGridBuilder()\
    .addGrid(linear_reg.maxIter, [10, 20])\
    .addGrid(linear_reg.regParam, [0.9, 0.7, 0.5, 0.3, 0.1]) \
    .addGrid(linear_reg.fitIntercept, [False, True])\
    .addGrid(linear_reg.elasticNetParam, [0.0, 0.3, 0.5, 0.8])\
    .build()


tvs = TrainValidationSplit(estimator=linear_reg,
                           estimatorParamMaps=param_Grid,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.8)

lr_Model = linear_reg.fit(trainingData)

# COMMAND ----------

# Make predictions
predictions = lr_Model.transform(trainingData)
selected_cols = ["label", "prediction", "time_window"] 
predictions = predictions.select(selected_cols)
display(predictions)

# COMMAND ----------

import numpy as np
def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual))

yt = predictions.select("label").collect()
yp = predictions.select("prediction").collect()

MAPE(yt, yp)

# COMMAND ----------

trainingSummary = lr_Model.summary

print("MAE: %f" % trainingSummary.meanAbsoluteError)
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)

# COMMAND ----------

####  Gradient Boosted Trees Regressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

grad_boost_tree = GBTRegressor()

param_Grid = (ParamGridBuilder()
             .addGrid(grad_boost_tree.maxDepth, [2,4,6,10, 20])
             .addGrid(grad_boost_tree.maxIter, [10, 20, 40])
             .build())

tvs = TrainValidationSplit(estimator=grad_boost_tree,
                           estimatorParamMaps=param_Grid,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.8)

gbt_Model = grad_boost_tree.fit(trainingData)

# COMMAND ----------

# Make predictions
predictions = gbt_Model.transform(trainingData)
selected_cols = ["label", "prediction", "time_window"] 
predictions = predictions.select(selected_cols)
display(predictions)

# COMMAND ----------

import numpy as np
def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual))

yt = predictions.select("label").collect()
yp = predictions.select("prediction").collect()

MAPE(yt, yp)

# COMMAND ----------

regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = regression_evaluator.evaluate(predictions)
print("RMSE: %g" % rmse)
regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae = regression_evaluator.evaluate(predictions)
print("MAE: %g" % mae)
regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2")
r2 = regression_evaluator.evaluate(predictions)
print("r2: %g" % r2)

# COMMAND ----------

####  Random Forest Regressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import TrainValidationSplit
random_forest = RandomForestRegressor()

param_Grid = (ParamGridBuilder()
             .addGrid(random_forest.maxDepth, [4, 8, 12, 20])
             .addGrid(random_forest.maxBins, [10, 20, 60])
             .addGrid(random_forest.numTrees, [100, 500, 1000])
             .build())

tvs = TrainValidationSplit(estimator=random_forest,
                           estimatorParamMaps=param_Grid,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.8)

rf_Model = random_forest.fit(trainingData)

# COMMAND ----------

# Make predictions
predictions = rf_Model.transform(trainingData)
selected_cols = ["label", "prediction", "time_window"] 
predictions = predictions.select(selected_cols)
display(predictions)

# COMMAND ----------

import numpy as np
def MAPE(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual))

yt = predictions.select("label").collect()
yp = predictions.select("prediction").collect()

MAPE(yt, yp)

# COMMAND ----------

regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = regression_evaluator.evaluate(predictions)
print("RMSE: %g" % rmse)
regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="mae")
mae = regression_evaluator.evaluate(predictions)
print("MAE: %g" % mae)
regression_evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="r2")
r2 = regression_evaluator.evaluate(predictions)
print("r2: %g" % r2)
