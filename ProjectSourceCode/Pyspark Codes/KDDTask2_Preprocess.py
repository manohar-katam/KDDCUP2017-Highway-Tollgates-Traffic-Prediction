# Databricks notebook source
import pandas as pd
import cStringIO
from pyspark.sql import *
loadData = sc.wholeTextFiles("/FileStore/tables/kdpnf22g1500431414483/volume_table_6__training-4f549.csv").collect()[0][1]
output = cStringIO.StringIO(loadData)
volumeData = pd.read_csv(output)

loadData = sc.wholeTextFiles("/FileStore/tables/xjqrdn561501441249853/volume_table_6__test1-ca2ed.csv").collect()[0][1]
output = cStringIO.StringIO(loadData)
volumeDataTest = pd.read_csv(output)

loadData = sc.wholeTextFiles("/FileStore/tables/kdpnf22g1500431414483/weather__table_7__training_update-ae54a.csv").collect()[0][1]
output = cStringIO.StringIO(loadData)
weatherData = pd.read_csv(output)

loadData = sc.wholeTextFiles("/FileStore/tables/xjqrdn561501441249853/weather__table_7__test1-781c6.csv").collect()[0][1]
output = cStringIO.StringIO(loadData)
weatherDataTest = pd.read_csv(output)

# COMMAND ----------

print weatherData.shape, weatherDataTest.shape

# COMMAND ----------

## append test weather data
weatherData = weatherData.append(weatherDataTest).reset_index()

# COMMAND ----------

weatherData.head()

# COMMAND ----------

weatherData.shape

# COMMAND ----------

# replacing the outlier value of 99017 in wind_direction of weatherData by avg of previous and next value
for i, row in weatherData.iterrows():
    if row['wind_direction']== 999017.0:
        previous_value = weatherData.loc[i-1,'wind_direction']
        next_value = weatherData.loc[i+1,'wind_direction']
        if next_value != 999017.0:
            weatherData.loc[i, 'wind_direction'] = (previous_value + next_value)/2.0
        else:
            weatherData.loc[i, 'wind_direction'] = previous_value

# COMMAND ----------

volumeData.head()

# COMMAND ----------

## since vehicle model and vehicle type are just model and type of vehicles passed by tollgate we don't require those features
## also, has_etc has nothing to do with volume, so we also removed that feature
volumeData['time'] =  pd.to_datetime(volumeData['time'] , format='%Y-%m-%d %H:%M:%S')
volumeData = volumeData.set_index(['time'])
# group by 20 minutes and finding the volume of traffic passing
volumeDataCorr = volumeData.groupby([pd.TimeGrouper('20Min'), 'tollgate_id', 'direction', 'vehicle_model','has_etc','vehicle_type']).size()\
       .reset_index().rename(columns = {0:'volume'})
# volumeDataPlus5 = volumeData.groupby([pd.TimeGrouper(freq='20Min',base=5, label='right') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataMinus5 = volumeData.groupby([pd.TimeGrouper(freq='20Min',base=55, label='left') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataPlus3 = volumeData.groupby([pd.TimeGrouper(freq='20Min',base=3, label='right') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataMinus3 = volumeData.groupby([pd.TimeGrouper(freq='20Min',base=57, label='left') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
volumeData = volumeData.groupby([pd.TimeGrouper('20Min'), 'tollgate_id', 'direction']).size()\
       .reset_index().rename(columns = {0:'volume'})



# COMMAND ----------

volumeData

# COMMAND ----------

#volumeData = volumeData.append(volumeDataPlus5).append(volumeDataMinus5).append(volumeDataPlus3).append(volumeDataMinus3)

# COMMAND ----------

#import seaborn as sns
#sns.heatmap(volumeDataCorr.corr(), annot=True, fmt=".2f")
#display()

# COMMAND ----------

volumeDataTest['time'] =  pd.to_datetime(volumeDataTest['time'] , format='%Y-%m-%d %H:%M:%S')
volumeDataTest = volumeDataTest.set_index(['time'])

# group by 20 minutes and finding the volume of traffic passing
# volumeDataTestPlus5 = volumeDataTest.groupby([pd.TimeGrouper(freq='20Min',base=5, label='right') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataTestMinus5 = volumeDataTest.groupby([pd.TimeGrouper(freq='20Min',base=55, label='left') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataTestPlus3 = volumeDataTest.groupby([pd.TimeGrouper(freq='20Min',base=3, label='right') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
# volumeDataTestMinus3 = volumeDataTest.groupby([pd.TimeGrouper(freq='20Min',base=57, label='left') , 'tollgate_id', 'direction']).size()\
#        .reset_index().rename(columns = {0:'volume'})
volumeDataTest = volumeDataTest.groupby([pd.TimeGrouper('20Min'), 'tollgate_id', 'direction']).size()\
       .reset_index().rename(columns = {0:'volume'})

# COMMAND ----------

volumeDataTest.head()

# COMMAND ----------

#volumeDataTest = volumeDataTest.append(volumeDataTestPlus5).append(volumeDataTestMinus5).append(volumeDataTestPlus3).append(volumeDataTestMinus3)

# COMMAND ----------

print volumeData.shape, volumeDataTest.shape

# COMMAND ----------

# append the volume test data to volume data
volumeData = volumeData.append(volumeDataTest)

# COMMAND ----------

## lag the volume feature
volumeData['lag1'] = volumeData['volume'].shift(1)
volumeData['lag2'] = volumeData['volume'].shift(2)
volumeData['lag3'] = volumeData['volume'].shift(3)
volumeData['lag4'] = volumeData['volume'].shift(4)
volumeData['lag5'] = volumeData['volume'].shift(5)
volumeData['lag6'] = volumeData['volume'].shift(6)
volumeData['lag7'] = volumeData['volume'].shift(7)

# COMMAND ----------

volumeData

# COMMAND ----------

import seaborn as sns
#sns.heatmap(volumeData.corr(), annot=True, fmt=".2f")
#display()

# COMMAND ----------

volumeDataTest['time'] = volumeDataTest['time'] + pd.DateOffset(hours=2)
volumeDataTest2 = volumeDataTest
del volumeDataTest

# COMMAND ----------

volumeDataTest2.head()

# COMMAND ----------

print volumeData.shape, volumeDataTest2.shape

# COMMAND ----------

# append columns to find whether it is a holiday, weekend or weekday and what is the hour in 24 hours
from datetime import datetime
start_date1 = datetime(2016, 9, 15)  ## 9/15 to 9/17 Mid autumn festival holidays: reference: http://www.officeholidays.com/countries/china/2016.php
end_date1 = datetime(2016, 9, 17)

start_date2 = datetime(2016, 10, 1)  ## Oct 1st to 7th are National Holidays, so we will take this range into account
end_date2 = datetime(2016, 10, 7)

rnge = pd.date_range(start_date1, end_date1).append(pd.date_range(start_date2, end_date2))

def imputeDateData(Data,Time):
    
    # hour, weekday, timewindow
    hour = pd.get_dummies(Data[Time].dt.hour,prefix='hour_')
    weekday = pd.get_dummies(Data[Time].dt.weekday_name)
    minute= pd.get_dummies(Data[Time].dt.minute)
    # conatenate to Data
    Data = pd.concat([Data,weekday,hour, minute], axis=1)
    # the date hour feature makes it easy to pair or add weather data
    Data['date']=Data[Time].dt.date
    Data['date'] = Data['date'].astype(str)
    Data['date'] = pd.to_datetime(Data['date'], format='%Y-%m-%d')
    Data['hour']=Data[Time].dt.hour.astype(int)
    # whether the day is a holiday
    date = Data[Time].dt.date
    for i, row in Data.iterrows():
        Data.loc[i, "holiday"] = 0
        if date.loc[i] in rnge: Data.loc[i, "holiday"] = 1
    return Data

# COMMAND ----------

volumeData = imputeDateData(volumeData,'time')
volumeDataTest2 = imputeDateData(volumeDataTest2,'time')

# COMMAND ----------

volumeData.head()

# COMMAND ----------

print volumeData.shape, volumeDataTest2.shape

# COMMAND ----------

# Add weather data by date and time
weatherData['date'] = pd.to_datetime(weatherData['date'], format='%Y-%m-%d')

# COMMAND ----------

# Turn hour into 3 hour intervals and then combine with weather data
def addWeatherData(Data):
    for i, row in Data.iterrows():
        if row['hour'] in [23,0,1]: Data.loc[i, "hour"] = 0
        elif row['hour'] in [2,3,4]: Data.loc[i, "hour"] = 3 
        elif row['hour'] in [5,6,7]: Data.loc[i, "hour"] = 6         
        elif row['hour'] in [8,9,10]: Data.loc[i, "hour"] = 9         
        elif row['hour'] in [11,12,13]: Data.loc[i, "hour"] = 12         
        elif row['hour'] in [14,15,16]: Data.loc[i, "hour"] = 15         
        elif row['hour'] in [17,18,19]: Data.loc[i, "hour"] = 18         
        elif row['hour'] in [20,21,22]: Data.loc[i, "hour"] = 21
    return pd.merge(Data,weatherData,on =['date', 'hour'] ,how='left')

# COMMAND ----------

volumeData = addWeatherData(volumeData)
volumeDataTest2 = addWeatherData(volumeDataTest2)

# COMMAND ----------

volumeData.head()

# COMMAND ----------

print volumeData.shape, volumeDataTest2.shape

# COMMAND ----------

volumeData = volumeData.drop(['hour','date'],axis=1)
volumeDataTest2 = volumeDataTest2.drop(['hour','date'],axis=1)

# COMMAND ----------

volumeData.head()

# COMMAND ----------

volumeData['end'] = volumeData['time'] + pd.DateOffset(minutes=20)
volumeDataTest2['end'] = volumeDataTest2['time'] + pd.DateOffset(minutes=20)

# COMMAND ----------

volumeData.head()

# COMMAND ----------

def createTimeWindows(Data,start,end):
    strt = Data[start].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    en = Data[end].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    Data['time_window'] = '['+ strt +','+ en +')'
    return Data.drop([start,end],axis =1)

# COMMAND ----------

volumeData = createTimeWindows(volumeData,'time','end')
volumeDataTest2 = createTimeWindows(volumeDataTest2,'time','end')

# COMMAND ----------

volumeData.head()

# COMMAND ----------

volumeData = volumeData.set_index(['tollgate_id','time_window', 'direction'])
volumeDataTest2 = volumeDataTest2.set_index(['tollgate_id','time_window', 'direction'])

# COMMAND ----------

volumeData

# COMMAND ----------

volume_testcol,volume_traincol = list(volumeDataTest2.columns.values),list(volumeData.columns.values)

missingData=  [x for x in volume_traincol  if x not in volume_testcol]

for label in missingData:
    volumeDataTest2[label] = 0
    
volumeDataTest2 = volumeDataTest2[volume_traincol]

# COMMAND ----------

print volumeData.shape, volumeDataTest2.shape

# COMMAND ----------

## fill any missed NA values with mean of the column
def fillWithMean(data):
    return data.fillna(data.mean())

volumeData =fillWithMean(volumeData)
volumeDataTest2 = fillWithMean(volumeDataTest2)

# COMMAND ----------

print volumeData.shape, volumeDataTest2.shape

# COMMAND ----------

volumeData.head()

# COMMAND ----------

weathers = ['pressure', 'sea_pressure', 'wind_direction', 'wind_speed', 'temperature', 'rel_humidity', 'precipitation', 'volume']

# COMMAND ----------

import seaborn as sns2
sns2.heatmap(volumeData[weathers].corr(), annot=True, fmt=".2f")
display()

# COMMAND ----------

days = ['Monday', 'Wednesday', 'Tuesday', 'Thursday', 'Friday', 'Saturday', 'Sunday','holiday', 'volume']

# COMMAND ----------

import seaborn as sns3
sns3.heatmap(volumeData[days].corr(), annot=True, fmt=".2f")
display()

# COMMAND ----------

# save the pre-processed file to csv
volumeData.to_csv('/dbfs/FileStore/tables/preprocessed_training_data_task2.csv')
volumeDataTest2.to_csv('/dbfs/FileStore/tables/preprocessed_test_data_task2.csv')

# COMMAND ----------


