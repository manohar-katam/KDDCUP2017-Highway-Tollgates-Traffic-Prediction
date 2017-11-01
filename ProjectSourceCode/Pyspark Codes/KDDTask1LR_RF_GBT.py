# Databricks notebook source
task1Data = spark.read.option("header","true").option("inferSchema","true").csv("/FileStore/tables/preprocessed_training_data_task1.csv")
task1Data.head()

# COMMAND ----------

display(task1Data)

# COMMAND ----------

task1Data.printSchema()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

cols = task1Data.columns

# COMMAND ----------

task1Data.createOrReplaceTempView("data")

# COMMAND ----------

data_F = spark.sql("select `intersection_id`,	`tollgate_id`,	`time_window`,averagetravltime,lag1,lag2,lag3,lag4,lag5,lag6,lag7,Friday,	Monday,	Saturday,	Sunday,	Thursday,Tuesday,Wednesday,	hour__0,	hour__1,	hour__2,	hour__3,	hour__4,	hour__5,	hour__6,	hour__7,	hour__8,	hour__9,	hour__10,	hour__11,	hour__12,	hour__13,	hour__14,	hour__15,	hour__16,	hour__17,	hour__18,	hour__19,	hour__20,	hour__21,	hour__22,	hour__23,	0,	20,	40,	China_hollidays,	index,	pressure,	sea_pressure,	wind_direction,	wind_speed,	temperature,	rel_humidity,	precipitation,	inlink_crscount,	outlink_crscount,	length,	linkcnt,	lane1_length,	lane2_length,	lane3_length,lane4_length,	lane1_count,lane2_count,lane3_count,lane4_count , averagetravltime as label from data")
display(data_F)


# COMMAND ----------



from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

cat_Colns = ["intersection_id", "time_window"]
stges = [] 
for cat_Coln in cat_Colns:
 
  string_Indexer = StringIndexer(inputCol=cat_Coln, outputCol=cat_Coln+"Index")
 
  en_coder = OneHotEncoder(inputCol=cat_Coln+"Index", outputCol=cat_Coln+"classVec")
 
  stges += [string_Indexer, en_coder]

# COMMAND ----------


numeric_Cols = ["tollgate_id", "Friday",	"Monday",	"Saturday",	"Sunday",	"Thursday",	"Tuesday",	"Wednesday","lag1","lag2","lag3","lag4","lag5","lag6","lag7",	"hour__0",	"hour__1",	"hour__2",	"hour__3",	"hour__4",	"hour__5",	"hour__6",	"hour__7",	"hour__8",	"hour__9",	"hour__10",	"hour__11",	"hour__12",	"hour__13",	"hour__14",	"hour__15",	"hour__16",	"hour__17",	"hour__18",	"hour__19",	"hour__20",	"hour__21",	"hour__22",	"hour__23",	"0",	"20",	"40",	"China_hollidays",	"index",	"pressure",	"sea_pressure",	"wind_direction",	"wind_speed",	"temperature",	"rel_humidity",	"precipitation",	"inlink_crscount",	"outlink_crscount",	"length",	"linkcnt",	"lane1_length",	"lane2_length",	"lane3_length",	"lane4_length",	"lane1_count",	"lane2_count",	"lane3_count",	"lane4_count"]
assembler_Input = map(lambda c: c + "classVec", cat_Colns) + numeric_Cols
assembler = VectorAssembler(inputCols=assembler_Input, outputCol="features")
stges += [assembler]

# COMMAND ----------


pipeline = Pipeline(stages=stges)
pipelineModel = pipeline.fit(data_F)
data_F = pipelineModel.transform(data_F)
select_cols = ["label", "features"] + cols
data_F = data_F.select(select_cols)
display(data_F)

# COMMAND ----------

trainingData = data_F
print trainingData.count()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
linear_reg = LinearRegression()
param_Grd = ParamGridBuilder().addGrid(linear_reg.maxIter, [10, 20]).addGrid(linear_reg.regParam, [0.9, 0.7, 0.5, 0.3, 0.1]) \
    .addGrid(linear_reg.fitIntercept, [False, True]).addGrid(linear_reg.elasticNetParam, [0.0, 0.5, 1.0]).build()
t_v_s = TrainValidationSplit(estimator=linear_reg,
                           estimatorParamMaps=param_Grd,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.8)
lrModel = linear_reg.fit(trainingData)

# COMMAND ----------


pred = lrModel.transform(trainingData)
selected_cols = ["label", "prediction", "time_window"] 
pred = pred.select(selected_cols)
display(pred)

# COMMAND ----------

import numpy as ny
def Mean_Absolute_Percentage_Error(labl,predction):
    labl, predction = ny.array(labl), ny.array(predction)
    return ny.mean(ny.abs((labl - predction) / labl))
y_label = pred.select("label").collect()
y_predicton = pred.select("prediction").collect()
Mean_Absolute_Percentage_Error(y_label, y_predicton)

# COMMAND ----------

trainingSummary = lrModel.summary
print("Mean_Absolute_Error:(MAE) %f" % trainingSummary.meanAbsoluteError)
print("Root_Mean_Squared_Error(RMSE): %f" % trainingSummary.rootMeanSquaredError)
print("Root_squared r2: %f" % trainingSummary.r2)

# COMMAND ----------

####  Gradient Boosted Trees Regressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

gGradient_boost = GBTRegressor()

param_Grd = (ParamGridBuilder()
             .addGrid(gGradient_boost.maxDepth, [2,4,6,10, 20])
             .addGrid(gGradient_boost.maxIter, [10, 20, 40])
             .build())

t_v_s = TrainValidationSplit(estimator=gGradient_boost,
                           estimatorParamMaps=param_Grd,
                           evaluator=RegressionEvaluator(),
                           trainRatio=0.8)

gbtModel = gGradient_boost.fit(trainingData)


# COMMAND ----------

pred = gbtModel.transform(trainingData)
selcted_cols = ["label", "prediction", "time_window"] 
pred = pred.select(selcted_cols)
display(pred)

# COMMAND ----------

import numpy as ny
def Mean_Absolute_Percentage_Error(labl,predction):
    labl, predction = ny.array(labl), ny.array(predction)
    return ny.mean(ny.abs((labl - predction) / labl))
y_label = pred.select("label").collect()
y_predicton = pred.select("prediction").collect()
Mean_Absolute_Percentage_Error(y_label, y_predicton)

# COMMAND ----------

evalator = RegressionEvaluator( labelCol="label", predictionCol="prediction", metricName="mae")
mean_absolute_error = evalator.evaluate(pred)
print("mean_absolute_error(MAE): %g" % mean_absolute_error)
evalator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
root_mean_squared_error = evalator.evaluate(pred)
print("root_mean_squared_error(RMSE): %g" % root_mean_squared_error)
evalator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
root_squared = evalator.evaluate(pred)
print("root_squared(r2): %g" % root_squared)

# COMMAND ----------


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

random_forest = RandomForestRegressor()

param_Grd = (ParamGridBuilder().addGrid(random_forest.maxDepth, [2, 4, 6, 8]).addGrid(random_forest.maxBins, [20, 60])
             .addGrid(random_forest.numTrees, [5, 20, 50, 100]) .build())

t_v_s = TrainValidationSplit(estimator=random_forest,estimatorParamMaps=param_Grd,evaluator=RegressionEvaluator(),
                           trainRatio=0.8)

rfModel = random_forest.fit(trainingData)


# COMMAND ----------

pred = rfModel.transform(trainingData)
select_cols = ["label", "prediction", "time_window"] 
pred = pred.select(select_cols)
display(pred)

# COMMAND ----------

import numpy as ny
def Mean_Absolute_Percentage_Error(labl,predction):
    labl, predction = ny.array(labl), ny.array(predction)
    return ny.mean(ny.abs((labl - predction) / labl))
y_label = pred.select("label").collect()
y_predicton = pred.select("prediction").collect()
Mean_Absolute_Percentage_Error(y_label, y_predicton)

# COMMAND ----------

evalator = RegressionEvaluator( labelCol="label", predictionCol="prediction", metricName="mae")
mean_absolute_error = evalator.evaluate(pred)
print("mean_absolute_error(MAE): %g" % mean_absolute_error)
evalator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
root_mean_squared_error = evalator.evaluate(pred)
print("root_mean_squared_error(RMSE): %g" % root_mean_squared_error)
evalator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
root_squared = evalator.evaluate(pred)
print("root_squared(r2): %g" % root_squared)
