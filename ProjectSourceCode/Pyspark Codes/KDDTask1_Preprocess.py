# Databricks notebook source
# Loading all the data from csvfiles
import pandas as ps
from copy import copy
import cStringIO
table5_train = sc.wholeTextFiles("/FileStore/tables/urq6xroz1501511577436/trajectories_table_5__training-2c00c.csv").collect()[0][1]
table5_output = cStringIO.StringIO(table5_train)
trajectoryData_train = ps.read_csv(table5_output)
table5_test = sc.wholeTextFiles("/FileStore/tables/urq6xroz1501511577436/trajectories_table_5__test1-c6820.csv").collect()[0][1]
table5_test_output = cStringIO.StringIO(table5_test)
trajectoryData_test = ps.read_csv(table5_test_output)
Sample_Data = sc.wholeTextFiles("/FileStore/tables/r0alxf8v1501640905637/submission_sample_travelTime.csv").collect()[0][1]
Sample_Data_output = cStringIO.StringIO(Sample_Data)
Sample_Data_final= ps.read_csv(Sample_Data_output)
table4_train = sc.wholeTextFiles("/FileStore/tables/4irp0f0r1501803230304/routes__table_4_-f8522.csv").collect()[0][1]
table4_train_output = cStringIO.StringIO(table4_train)
table4_route = ps.read_csv(table4_train_output)
table3_train = sc.wholeTextFiles("/FileStore/tables/4irp0f0r1501803230304/links__table_3_-87d30.csv").collect()[0][1]
table3_train_output = cStringIO.StringIO(table3_train)
table3_links = ps.read_csv(table3_train_output)
table7_train = sc.wholeTextFiles("/FileStore/tables/iq4jtr7z1501887583938/weather__table_7__training_update-ae54a.csv").collect()[0][1]
table7_train_output = cStringIO.StringIO(table7_train)
table7_weather = ps.read_csv(table7_train_output)
table7_test = sc.wholeTextFiles("/FileStore/tables/iq4jtr7z1501887583938/weather__table_7__test1-781c6.csv").collect()[0][1]
table7_test_output = cStringIO.StringIO(table7_test)
table7_weather_test = ps.read_csv(table7_test_output)
print trajectoryData_train.shape,trajectoryData_test.shape,Sample_Data_final.shape,table4_route.shape,table3_links.shape,table7_weather.shape,table7_weather_test.shape

# COMMAND ----------

trajectoryData_train.head()

# COMMAND ----------

trajectoryData_train.describe()

# COMMAND ----------

#Replacing outlier values with the average of previous and next value
for k, row in trajectoryData_train.iterrows():
    if row['travel_time'] > 600:
        last_value = trajectoryData_train.loc[k-1,'travel_time']
        next_value = trajectoryData_train.loc[k+1,'travel_time']
        if last_value < 600:
            trajectoryData_train.loc[k, 'travel_time'] = (last_value + next_value)/2.0
        else:
            trajectoryData_train.loc[k, 'travel_time'] = last_value

# COMMAND ----------

for k, row in trajectoryData_test.iterrows():
    if row['travel_time'] > 600:
        last_value = trajectoryData_test.loc[k-1,'travel_time']
        next_value = trajectoryData_test.loc[k+1,'travel_time']
        if last_value < 600:
            trajectoryData_test.loc[k, 'travel_time'] = (last_value + next_value)/2.0
        else:
            trajectoryData_test.loc[k, 'travel_time'] = last_value

# COMMAND ----------

for k, row in trajectoryData_train.iterrows():
    if row['travel_time'] > 600:
        last_value = trajectoryData_train.loc[k-1,'travel_time']
        next_value = trajectoryData_train.loc[k+1,'travel_time']
        if last_value < 600:
            trajectoryData_train.loc[k, 'travel_time'] = (last_value + next_value)/2.0
        else:
            trajectoryData_train.loc[k, 'travel_time'] = last_value

# COMMAND ----------

trajectoryData_train.describe()

# COMMAND ----------

trajectoryData_train['starting_time'] = ps.to_datetime(trajectoryData_train['starting_time'], format='%Y-%m-%d %H:%M:%S')
trajectoryData_train = trajectoryData_train.set_index(['starting_time'])
trajectoryData_train = trajectoryData_train.groupby([ps.TimeGrouper('20Min'), 'intersection_id', 'tollgate_id']).travel_time.mean().reset_index().rename(columns={'travel_time':'averagetravltime'})
trajectoryData_test['starting_time'] = ps.to_datetime(trajectoryData_test['starting_time'], format="%Y-%m-%d %H:%M:%S")
trajectoryData_test = trajectoryData_test.set_index(['starting_time'])
trajectoryData_test = trajectoryData_test.groupby([ps.TimeGrouper('20Min'), 'intersection_id', 'tollgate_id']).travel_time.mean().reset_index().rename(columns={'travel_time':'averagetravltime'})
print trajectoryData_train.shape,trajectoryData_test.shape

# COMMAND ----------

trajectoryData_train.head()

# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

all_toll_intersections = []
for j in range(Sample_Data_final.shape[0]):
    
    intersection=Sample_Data_final.loc[j]['intersection_id']
    tollgate=Sample_Data_final.loc[j]['tollgate_id']
    token = (intersection,tollgate)
    if token not in all_toll_intersections:
        all_toll_intersections.append(token)
Sample_time = []
Sample_times = Sample_Data_final[(Sample_Data_final['tollgate_id']==1)&(Sample_Data_final['intersection_id']=='B') ]['time_window']
for st in Sample_times:
    Sample_time.append(ps.to_datetime(st.split(',')[0][1:], format="%Y-%m-%d %H:%M:%S") - ps.DateOffset(hours=2))
Sample_time = ps.Series(Sample_time).values

# COMMAND ----------

def replace_missing_time(test,tollgate,intersection,iteration,Sample_time):
    while iteration > 0:
        try:
            missing_time = test[(test['tollgate_id']==tollgate) & (test['starting_time'] == Sample_time[iteration - 1])& (test['intersection_id']==intersection) ]['averagetravltime']
            return missing_time.values[0]
        except Exception,e:
            iteration = iteration - 1
            continue
       

# COMMAND ----------


for intersection, tollgate in all_toll_intersections:
    test_toll_intersections = copy(trajectoryData_test[(trajectoryData_test['tollgate_id']==tollgate) & (trajectoryData_test['intersection_id']==intersection) ].reset_index()) 
    test_time= trajectoryData_test[(trajectoryData_test['tollgate_id']==tollgate) & (trajectoryData_test['intersection_id']==intersection)]['starting_time'].values
    test_toll_intersections.drop('index',axis=1,inplace=True)
    test_toll_intersections = test_toll_intersections.loc[0]
    for k in range(len(Sample_time)):
        if Sample_time[k] not in test_time: 
            test_toll_intersections['starting_time'] = Sample_time[k]
            test_toll_intersections['averagetravltime'] = replace_missing_time(trajectoryData_test, tollgate, intersection,k, Sample_time)
            trajectoryData_test = trajectoryData_test.append(test_toll_intersections)
trajectoryData_test = trajectoryData_test.reset_index()
trajectoryData_test.drop('index', axis=1, inplace=True)


# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

trajectoryData_train = trajectoryData_train.append(trajectoryData_test)

# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

## adding lag features
trajectoryData_train['lag1'] = trajectoryData_train['averagetravltime'].shift(1)
trajectoryData_train['lag2'] = trajectoryData_train['averagetravltime'].shift(2)
trajectoryData_train['lag3'] = trajectoryData_train['averagetravltime'].shift(3)
trajectoryData_train['lag4'] = trajectoryData_train['averagetravltime'].shift(4)
trajectoryData_train['lag5'] = trajectoryData_train['averagetravltime'].shift(5)
trajectoryData_train['lag6'] = trajectoryData_train['averagetravltime'].shift(6)
trajectoryData_train['lag7'] = trajectoryData_train['averagetravltime'].shift(7)

# COMMAND ----------

import seaborn as sns
sns.heatmap(trajectoryData_train.corr(), annot = True, fmt = ".2f")
display()

# COMMAND ----------

trajectoryData_test['starting_time'] = trajectoryData_test['starting_time'] + ps.DateOffset(hours=2)
trajectoryData_test_duplicate=trajectoryData_test
trajectoryData_test_duplicate.drop('averagetravltime',axis=1,inplace=True)

# COMMAND ----------

trajectoryData_test_duplicate.shape
trajectoryData_train.shape

# COMMAND ----------

trajectoryData_train.head()

# COMMAND ----------

# Chinese main festival days
from datetime import datetime
start_date = datetime(2016, 9, 15)
end_date = datetime(2016, 9, 17)
Holiday_range = ps.date_range(start_date, end_date)
start_date2 = datetime(2016, 10, 1)
end_date2 = datetime(2016, 10, 7)
Holiday_range= Holiday_range.append(ps.date_range(start_date2, end_date2))

# COMMAND ----------

# Adding Extra column with the name China_holidays. If the date exists in between Holiday range,the value of the column will be 1 or else 0
def Identify_holiday_dates(Data,start_time):
    Day_of_the_week = ps.get_dummies(Data[start_time].dt.weekday_name)
    hr_of_the_day = ps.get_dummies(Data[start_time].dt.hour, prefix='hour_')
    minute= ps.get_dummies(Data[start_time].dt.minute)
    Data = ps.concat([Data,Day_of_the_week,hr_of_the_day,minute], axis=1)
    Data['date']=Data[start_time].dt.date
    Data['date'] = Data['date'].astype(str)
    Data['date'] = ps.to_datetime(Data['date'], format='%Y-%m-%d')
    Data['hour']=Data[start_time].dt.hour.astype(int)
    start_time_date = Data[start_time].dt.date
    for k, row in Data.iterrows():
        Data.loc[k,"China_hollidays"] = 0
        if start_time_date.loc[k] in Holiday_range: Data.loc[k, "China_hollidays"] = 1
    return Data

# COMMAND ----------

trajectoryData_train = Identify_holiday_dates(trajectoryData_train, "starting_time")
trajectoryData_test_duplicate = Identify_holiday_dates(trajectoryData_test_duplicate, "starting_time")

# COMMAND ----------

trajectoryData_test_duplicate.shape
trajectoryData_train.head()

# COMMAND ----------

# loading and appending weather test data to weather train data


table7_weather = table7_weather.append(table7_weather_test).reset_index()
table7_weather['date'] = ps.to_datetime(table7_weather['date'], format='%Y-%m-%d')

# COMMAND ----------

# replacing the outlier value of 99017 in wind_direction of weatherData by avg of previous and next value
for i, row in table7_weather.iterrows():
    if row['wind_direction']== 999017.0:
        previous_value = table7_weather.loc[i-1,'wind_direction']
        next_value = table7_weather.loc[i+1,'wind_direction']
        if next_value != 999017.0:
            table7_weather.loc[i, 'wind_direction'] = (previous_value + next_value)/2.0
        else:
            table7_weather.loc[i, 'wind_direction'] = previous_value

# COMMAND ----------

table7_weather.head()

# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

# Turn hour into 3 hour intervals and then combine with weather data
def addWeatherData(df):
    for i, row in df.iterrows():
        if row['hour'] in [23,0,1]: df.loc[i, "hour"] = 0
        elif row['hour'] in [2,3,4]: df.loc[i, "hour"] = 3 
        elif row['hour'] in [5,6,7]: df.loc[i, "hour"] = 6         
        elif row['hour'] in [8,9,10]: df.loc[i, "hour"] = 9         
        elif row['hour'] in [11,12,13]: df.loc[i, "hour"] = 12         
        elif row['hour'] in [14,15,16]: df.loc[i, "hour"] = 15         
        elif row['hour'] in [17,18,19]: df.loc[i, "hour"] = 18         
        elif row['hour'] in [20,21,22]: df.loc[i, "hour"] = 21
    return ps.merge(df,table7_weather,on =['date', 'hour'] ,how='left')

# COMMAND ----------

trajectoryData_train = addWeatherData(trajectoryData_train)
trajectoryData_test_duplicate = addWeatherData(trajectoryData_test_duplicate)


# COMMAND ----------

trajectoryData_train.head()

# COMMAND ----------

trajectoryData_train = trajectoryData_train.drop(['hour','date'],axis=1)
trajectoryData_test_duplicate = trajectoryData_test_duplicate.drop(['hour','date'],axis=1)
trajectoryData_train.shape

# COMMAND ----------


divison_row = []
def divide(srng): return srng.split(',')
table4_route.link_seq = table4_route.link_seq.apply(divide)

_ = table4_route.apply(lambda row: [divison_row.append([row['intersection_id'], row['tollgate_id'], link]) 
                         for link in row.link_seq], axis=1)
table_headers = ['intersection_id', 'tollgate_id', 'link_id']
table4_route_new= ps.DataFrame(divison_row, columns=table_headers)
table4_route_new['link_id'] = table4_route_new['link_id'].astype(str)


# COMMAND ----------

table3_links['crsin'] = 0
table3_links['crsout'] = 0
for k, row in table3_links.iterrows():
    if ',' in str(row['out_top']):
        table3_links.loc[k, 'crsout'] = 1
    if ',' in str(row['in_top']):
        table3_links.loc[k, 'crsin'] = 1
table3_links['link_id'] = table3_links['link_id'].astype(str)  
table4_route_new = ps.merge(table4_route_new,table3_links, on='link_id', how='left')
table4_route_new.drop(['in_top', 'out_top'], axis=1, inplace=True)

# COMMAND ----------

join_incount= table4_route_new[['intersection_id', 'tollgate_id', 'crsin']].groupby(['intersection_id', 'tollgate_id'])\
               .crsin.sum().reset_index().rename(columns={'crsin':'inlink_crscount'})
join_outcount = table4_route_new[['intersection_id', 'tollgate_id', 'crsout']].groupby(['intersection_id', 'tollgate_id'])\
               .crsout.sum().reset_index().rename(columns={'crsout':'outlink_crscount'})
final = ps.merge(join_incount,join_outcount,on=['intersection_id', 'tollgate_id'],how='left')
len= table4_route_new[['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum().reset_index()
final = ps.merge(final, len, on=['intersection_id', 'tollgate_id'], how='left')
linkcnt = table4_route_new[['intersection_id', 'tollgate_id']] .groupby(['intersection_id', 'tollgate_id']).size()\
        .reset_index().rename(columns={0:'linkcnt'})
final = ps.merge(final, linkcnt, on=['intersection_id', 'tollgate_id'], how='left')
lane1_length = table4_route_new[table4_route_new.lanes==1][['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum()\
        .reset_index().rename(columns={'length':'lane1_length'})
final = ps.merge(final, lane1_length, on=['intersection_id', 'tollgate_id'],how='left')
lane1_count = table4_route_new[table4_route_new.lanes== 1][['intersection_id', 'tollgate_id']].groupby(['intersection_id', 'tollgate_id']).size()\
    .reset_index().rename(columns = {0:'lane1_count'})
final = ps.merge(final,lane1_count,on =['intersection_id', 'tollgate_id'] ,how='left')
lane2_length = table4_route_new[table4_route_new.lanes==2][['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum()\
        .reset_index().rename(columns={'length':'lane2_length'})
final = ps.merge(final, lane2_length, on=['intersection_id', 'tollgate_id'],how='left')
lane2_count = table4_route_new[table4_route_new.lanes== 2][['intersection_id', 'tollgate_id']].groupby(['intersection_id', 'tollgate_id']).size()\
    .reset_index().rename(columns = {0:'lane2_count'})
final = ps.merge(final,lane2_count,on =['intersection_id', 'tollgate_id'] ,how='left')
lane3_length = table4_route_new[table4_route_new.lanes==3][['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum()\
        .reset_index().rename(columns={'length':'lane3_length'})
final = ps.merge(final, lane3_length, on=['intersection_id', 'tollgate_id'],how='left')
lane3_count = table4_route_new[table4_route_new.lanes== 3][['intersection_id', 'tollgate_id']].groupby(['intersection_id', 'tollgate_id']).size()\
    .reset_index().rename(columns = {0:'lane3_count'})
final = ps.merge(final,lane3_count,on =['intersection_id', 'tollgate_id'] ,how='left')
lane4_length = table4_route_new[table4_route_new.lanes==4][['intersection_id', 'tollgate_id', 'length']].groupby(['intersection_id', 'tollgate_id']).length.sum()\
        .reset_index().rename(columns={'length':'lane4_length'})
final = ps.merge(final, lane4_length, on=['intersection_id', 'tollgate_id'],how='left')
lane4_count = table4_route_new[table4_route_new.lanes== 4][['intersection_id', 'tollgate_id']].groupby(['intersection_id', 'tollgate_id']).size()\
    .reset_index().rename(columns = {0:'lane4_count'})
final = ps.merge(final,lane4_count,on =['intersection_id', 'tollgate_id'] ,how='left')
final.fillna(0, inplace=True)
trajectoryData_train = ps.merge(trajectoryData_train, final, on=['intersection_id', 'tollgate_id'], how='left')
trajectoryData_test_duplicate = ps.merge(trajectoryData_test_duplicate, final, on=['intersection_id', 'tollgate_id'], how='left')


# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

def timeperiod(Data, start_time, end_time):
    st = Data[start_time].apply(lambda k:k.strftime("%Y-%m-%d %H:%M:%S"))
    et = Data[end_time].apply(lambda k:k.strftime("%Y-%m-%d %H:%M:%S"))
    Data['time_window'] = '[' + st + ',' + et + ')'
    return Data.drop([start_time, end_time], axis=1)

# COMMAND ----------

trajectoryData_test_duplicate['end'] = trajectoryData_test_duplicate['starting_time'] + ps.DateOffset(minutes=20)
trajectoryData_train['end'] = trajectoryData_train['starting_time'] + ps.DateOffset(minutes=20)
trajectoryData_test_duplicate = timeperiod(trajectoryData_test_duplicate, 'starting_time', 'end')
trajectoryData_train = timeperiod(trajectoryData_train, 'starting_time', 'end')



# COMMAND ----------

trajectoryData_test_duplicate = trajectoryData_test_duplicate.set_index(['intersection_id','tollgate_id','time_window'])
trajectoryData_train = trajectoryData_train.set_index(['intersection_id','tollgate_id','time_window'])

# COMMAND ----------

trajectoryData_train.shape

# COMMAND ----------

trajectoryData_test_columns,trajectoryData_train_columns = list(trajectoryData_test_duplicate.columns.values),list(trajectoryData_train.columns.values)

# COMMAND ----------

mis5 =  [data for data in trajectoryData_train_columns  if data not in trajectoryData_test_columns]

# COMMAND ----------

for label in mis5:
    trajectoryData_test_duplicate[label] = 0

# COMMAND ----------

trajectoryData_test_duplicate = trajectoryData_test_duplicate[trajectoryData_train_columns]

# COMMAND ----------

def fill_nullvalues(data):
    return data.fillna(data.mean())
trajectoryData_test_duplicate = fill_nullvalues(trajectoryData_test_duplicate)

trajectoryData_train =fill_nullvalues(trajectoryData_train)


# COMMAND ----------

print trajectoryData_test_duplicate.shape, trajectoryData_train.shape

# COMMAND ----------

trajectoryData_test_duplicate.to_csv('/dbfs/FileStore/tables/preprocessed_test_data_task1.csv')
trajectoryData_train.to_csv('/dbfs/FileStore/tables/preprocessed_training_data_task1.csv')


# COMMAND ----------

trajectoryData_train.columns.values
