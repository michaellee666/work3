# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:01:02 2018

@author: Libo
"""

#math
import pandas as pd
import numpy as np
#plots
import matplotlib.pyplot as plt
import seaborn as sns

#from haversine import haversine

sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
spray = pd.read_csv('../input/spray.csv')
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
weather = pd.read_csv('../input/weather.csv')
#1.数据分析
print(len(train))
def datainspect(dataframe):
    print("missing values \n", dataframe.isnull().sum())
    print("dataframe index \n", dataframe.index)
    print("dataframe types \n", dataframe.dtypes)
    print("dataframe shape \n", dataframe.shape)
    print("dataframe describe \n", dataframe.describe())
    print("dataframe duplicates \n", dataframe.duplicated().sum())
    for item in dataframe:
        print(item)
        print(dataframe[item].nunique())
    return()

def drawlomap():
    mapdata = np.loadtxt("../input/mapdata_copyright_openstreetmap_contributors.txt")
    traps = pd.read_csv('../input/train.csv')[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    plt.figure(figsize=(10,14))
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, 
               aspect=aspect)

    traps1 = traps[traps.WnvPresent==1]
    traps0 = traps[traps.WnvPresent==0]

    traps1 = traps1[['Longitude', 'Latitude']].drop_duplicates().values
    traps0 = traps0[['Longitude', 'Latitude']].drop_duplicates().values

    plt.scatter(traps0[:,0], traps0[:,1], color='red', marker='*', alpha=1, label='Wnv = No');
    plt.scatter(traps1[:,0], traps1[:,1], color='blue', marker='*', alpha=1, label='Wnv = Yes');
    plt.legend();

    plt.savefig('trap_map.png');

#日期时间格式    
train['Date'] = pd.to_datetime(train['Date'])
spray['Date'] = pd.to_datetime(spray['Date'])
weather['Date'] = pd.to_datetime(weather['Date'])
#test['Date'] = pd.to_datetime(test['Date'])

#去除train表中多余的重复数据
train.drop(train[train.duplicated(keep='first')].index,axis=0,inplace=True)
print(len(train))

#去除spray表中多余的重复数据
spray.drop(spray[spray.duplicated(keep='first')].index,axis=0,inplace=True)

#weather表中缺失值处理
def rainy_day(column):
    weather[column] = weather[column].str.replace('T','0.005')
    weather[column] = weather[column].str.replace('M','0.0')
    weather[column] = weather[column].astype(float)
for col in ['Tavg','PrecipTotal','Depart','WetBulb','SnowFall',
            'StnPressure','SeaLevel','Depth','AvgSpeed','Heat','Cool']:
    rainy_day(col)
#weather表中Tavg缺失填充0后，需进行修改，赋值为Tmax和Tmin的均值。
weather['Tavg'][weather.Tavg==0] = (weather['Tmin'] + weather['Tmax']) / 2
#去除weather表中多余特征
cols = [col for col in weather.columns if col not in ('Station','Date')]
bad_col = []
for col in cols:
    try:
        weather[col] = pd.to_numeric(weather[col])
    except:
        bad_col.append(col)
weather.drop(bad_col,axis=1,inplace=True)
#train表中重复项去除
train = train.groupby(['Date',
                       'Address',
                       'Species',
                       'Block',
                       'Street',
                       'Trap',
                       'AddressNumberAndStreet',
                       'Latitude','Longitude',
                       'AddressAccuracy']).sum().reset_index()
train['WnvPresent'] = train['WnvPresent'].map(lambda x: 1 if x >= 1 else 0)
#train表中species特征的蚊子物种进行拆分
train = pd.get_dummies(train, columns=['Species'])

#weather表中Station经纬度
station1 = weather[weather['Station']==1].copy()
station2 = weather[weather['Station']==2].copy()
#Station 1: CHICAGO O'HARE INTERNATIONAL AIRPORT Lat: 41.995 Lon: -87.933 Elev: 662 ft. above sea level
station1['Latitude'] = 41.995
station1['Longitude'] = -87.9336
#Station 2: CHICAGO MIDWAY INTL ARPT Lat: 41.786 Lon: -87.752 Elev: 612 ft. above sea level
station2['Latitude'] = 41.78611
station2['Longitude'] = -87.75222
stations = pd.merge(station1,station2,on='Date',suffixes=('_s1','_s2'))
traps_weather = pd.merge(train,stations,on='Date')
#捕捉站到气象站的距离计算 
dist_1 = np.sqrt(((traps_weather['Latitude'] - traps_weather['Latitude_s1'])**2 + (traps_weather['Longitude'] - traps_weather['Longitude_s1'])**2))
dist_2 = np.sqrt(((traps_weather['Latitude'] - traps_weather['Latitude_s2'])**2 + (traps_weather['Longitude'] - traps_weather['Longitude_s2'])**2))
#计算每个捕捉站的距离权重，通过接近度来加权天气数据
total_dist = dist_1 + dist_2
traps_weather['weight_1'] = dist_1 / total_dist
traps_weather['weight_2'] = dist_2 / total_dist
#由于距离较近的站应该具有较重的权重，因此将距离权重应用于天气数据的反向权重。 
station1_list = [col for col in traps_weather.columns 
                if '_s1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]
station2_list = [col for col in traps_weather.columns 
                 if '_s2' in col and col not in ('Station_s2','Latitude_s2','Longitude_s2')]
for col in station1_list:
    traps_weather[col] = traps_weather['weight_2'] * traps_weather[col]
for col in station2_list:
    traps_weather[col] = traps_weather['weight_1'] * traps_weather[col]
#添加加权站1和站台2天气数据，并丢弃部分列。
for col in [col for col in traps_weather.columns 
            if 's1' in col and col not in ('Station_s1','Latitude_s1','Longitude_s1')]:
    name = col.replace('_s1','')
    traps_weather[name] = traps_weather[col] + traps_weather[name+'_s2']
    traps_weather.drop([col,name+'_s2'],axis=1,inplace=True)
#丢弃Station信息列
col1 = [col for col in traps_weather.columns if '_s1' in col]
col2 = [col for col in traps_weather.columns if '_s2' in col]
cols = col1 + col2
traps_weather.drop(cols,axis=1,inplace=True)


#目标干预的效果是在时间和距离两个维度上衰减。我们关心的参考位置是Trap，
distance = []
time = []
for i in traps_weather.index:
    temp_lat = traps_weather.at[i,'Latitude']
    temp_long = traps_weather.at[i,'Longitude']
    # calculate distance from traps to spray locations
    dist = np.sqrt((spray['Latitude'] - temp_lat)**2 + (spray['Longitude'] - temp_long)**2)
    distance.append(dist)
    # calculate time since spray
    time_since_spray = traps_weather.at[i,'Date'] - spray['Date']
    time_since_spray = time_since_spray.dt.total_seconds()
    time_since_spray = (((time_since_spray/60)/60)/24)
    time.append(time_since_spray)
distance = pd.DataFrame(distance)
time = pd.DataFrame(time)
time.reset_index(inplace=True)
time.drop('index',axis=1,inplace=True)
backup = time.copy()
for col in time.columns:
    time[col] = time[col].map(lambda x: 0 if x < 0 else x)
#时间和距离两个维度的喷雾衰减
spray_data = [time[i] * distance[i] for i in time.columns]
spray_data = pd.DataFrame(spray_data).transpose()
data = pd.merge(traps_weather,spray_data,how='inner',left_index=True,right_index=True)
#
def broadcasting_based(df):
    df_lats = np.array(df['Latitude'].tolist())
    df_longs = np.array(df['Longitude'].tolist())
    df_lats = np.deg2rad(df_lats)
    df_longs = np.deg2rad(df_longs) 

    df2_lats = np.array([41.995])
    df2_longs = np.array([-87.9336])
    df2_lats = np.deg2rad(df2_lats)
    df2_longs = np.deg2rad(df2_longs)
       
    diff_lat = df_lats - df2_lats
    diff_lng = df_longs - df2_longs

    d = np.sin(diff_lat/2)**2 + np.cos(df2_lats)*np.cos(df_lats) * np.sin(diff_lng/2)**2
    return 2 * 3959 * np.arcsin(np.sqrt(d))

distance_binary = broadcasting_based(spray)
#
from math import radians, cos, sin, asin, sqrt   
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）  
    """ 
    Calculate the great circle distance between two points  
    on the earth (specified in decimal degrees) 
    """  
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r * 1000 
print('calculating spray data...')
def distance_calc(i): 
    temp_lat = traps_weather.at[i,'Latitude']
    temp_long = traps_weather.at[i,'Longitude']
    # calculate distance from traps to spray locations
    dists = []
    if i % 500 == 0:
        print('distance '+str(i))
    for row in spray.itertuples(index=True,name=None):        
        dist = haversine(row[3],row[4],temp_lat,temp_long)
        dists.append(dist) 
    return dists
def time_calc(i): 
    if i % 500 == 0:
        print('time '+str(i))        
    # calculate time since spray
    time_since_spray = traps_weather.at[i,'Date'] - spray['Date']
    time_since_spray = time_since_spray.dt.total_seconds()
    time_since_spray = (((time_since_spray/60)/60)/24)    
    return time_since_spray
#num_cores = multiprocessing.cpu_count()
inputs = traps_weather.index
distance_binary = [distance_calc(i) for i in inputs[:2]]
#time_binary = [time_calc(i) for i in inputs]
#distance_binary = pd.DataFrame(distance_binary)
#time_binary = pd.DataFrame(time_binary)


print('starting...')
distance_binary = []
time_binary = []
for i in traps_weather.index:
    temp_lat = traps_weather.at[i,'Latitude']
    temp_long = traps_weather.at[i,'Longitude']
    # calculate distance from traps to spray locations
    dists = []
    if i % 500 == 0:
        print(i)
    for s in spray.index:
        dist = haversine(spray.at[s,'Latitude'],spray.at[s,'Longitude'],temp_lat,temp_long)
        dists.append(dist)
    distance_binary.append(dists)  
    # calculate time since spray
    time_since_spray = traps_weather.at[i,'Date'] - spray['Date']
    time_since_spray = time_since_spray.dt.total_seconds()
    time_since_spray = (((time_since_spray/60)/60)/24)
    time_binary.append(time_since_spray)

distance_binary = pd.DataFrame(distance_binary)
time_binary = pd.DataFrame(time_binary)
time_binary.reset_index(inplace=True)
time_binary.drop('index',axis=1,inplace=True)
# if observation took place before spray, zero out time
# else return elapsed time between spray and observation
for col in time_binary.columns:
    time_binary[col] = time_binary[col].map(lambda x: 0 if x < 0 else x)
time_binary.columns = distance_binary.columns
time_binary_backup = time_binary.copy()
distance_binary_backup = distance_binary.copy() 

time_tp = time_binary.transpose()
distance_tp = distance_binary.transpose()
binary = pd.merge(distance_tp,time_tp,how='inner',left_index=True,right_index=True,suffixes=('_d','_t'))
   
def CalculateDistance(i):
    distances = []
    if i % 500 == 0:
        print('evaluating time binaries '+str(i))
    d = i
    cols = distance_tp[[d]]

    if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
        
    if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)

    if len(cols[np.logical_and(cols[d] <= 0.5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 1,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    if len(cols[np.logical_and(cols[d] <= 5,cols[d] > 0)]) > 0:
        distances.append(1)
    else:
        distances.append(0)
    return distances

def CalculateTime(i):
    times = []
    if i % 500 == 0:
        print('evaluating time binaries '+str(i))

    t = i

    cols = time_tp[[t]]
    
    if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 7,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
        
    if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 30,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)

    if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    if len(cols[np.logical_and(cols[t] <= 90,cols[t] > 0)]) > 0:
        times.append(1)
    else:
        times.append(0)
    return times
    
print('binary bonanza...')
inputs = traps_weather.index
timevalues = [CalculateTime(i) for i in inputs[:2]]
distancevalues = [CalculateDistance(i) for i in inputs[:2]]

timevalues = pd.DataFrame(timevalues)
distancevalues = pd.DataFrame(distancevalues)
tv = timevalues.shape[1]
binary = pd.merge(distancevalues,timevalues,how='inner',left_index=True,right_index=True,suffixes=['_d','_t'])

for c in range(tv):
    binary[c] = binary[str(c)+'_d'] + binary[str(c)+'_t']
binary.rename(columns={0:'1week_halfmile',1:'1week_1mile',2:'1week_5mile',
                       3:'1month_halfmile',4:'1month_1mile',5:'1month_5mile',
                       6:'1quarter_halfmile',7:'1quarter_1mile',8:'1quarter_5mile'},inplace=True)
timecols = [col for col in binary.columns if '_t' in col]
distcols = [col for col in binary.columns if '_d' in col]
dropcols = timecols + distcols
binary.drop(dropcols,axis=1,inplace=True)
for col in binary.columns:
    binary[col] = binary[col].apply(lambda x: 1 if x == 2 else 0)
    
    
values = []
for i in traps_weather.index:
    observations = []
    if i % 500 == 0:
        print(i)
    d = str(i) + '_d'
    t = str(i) + '_t'
    
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
       np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] > 0),
           np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
       np.logical_and(binary[t] <= 7,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
        
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
           np.logical_and(binary[t] <= 30,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)

    if len(binary[np.logical_and(np.logical_and(binary[d] <= 0.5,binary[d] > 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 1,binary[d] >= 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    if len(binary[np.logical_and(np.logical_and(binary[d] <= 5,binary[d] > 0),
           np.logical_and(binary[t] <= 90,binary[t] > 0))]) > 0:
        observations.append(1)
    else:
        observations.append(0)
    values.append(observations)
    
values = pd.DataFrame(values)
values.to_csv('spray_data_binary.csv')
spray_col = ['1week_halfmile','1week_1mile','1week_5mile',
            '1month_halfmile','1month_1mile','1month_5mile',
            '1quarter_halfmile','1quarter_1mile','1quarter_5mile']
values.columns = spray_col
data = pd.merge(traps_weather,values,left_index=True,right_index=True)
data.to_csv('cleaned_data.csv')

#if __name__ == "__main__":    
#    datainspect(weather)#分析数据
#    drawlomap()