# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:00:05 2017

@author: colin

@edited by Vivien oct 2017

"""
#Wishlist:
## TODO: Combine with IBP rain gauge processing script
## TODO: Do comparisons

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PROCESS_wl_cruncher import find_csv_filenames
import re

root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\\'
raw_data_path = root_path + 'IBP Project\Data from Field Tablet\Rain Gauge Data\\'
NOAA_path = root_path + 'Regional Data\NOAA Precip\\'
NWIS_path = root_path + 'Regional Data\NWIS Precip\\'
processed_data_path = root_path + 'IBP Project\Documents\Deprecated Data Folder\Precipitation\\'

#For IBP rain gauge
files = find_csv_filenames(raw_data_path,suffix = '.csv')

# Gauge down; no rain or temperature data
down_data = [['2016-09-02 00:00:00','2017-04-26 00:00:00'], # Down due to error with optical datalogger link
             ['2017-09-25 00:00:00','2017-10-06 00:00:00']] # Down due to firmware failure (corrupted update)

# Gauge clogged, no rain but temperature still good.
clog_data = [['2017-06-08 00:00:00','2017-10-06 00:00:00'], # Clogged with excessive bird poop
             ['2018-08-02 16:00:00','2018-11-07 17:00:00']] # Tipping bucket stuck, clogged

def rg_combine():
    ts_main = pd.DataFrame({'date_time':[],
                        'temp_f':[],
                        'rain_in':[]
                        })
    for item in files:
        print item
        ts_raw=pd.read_table(raw_data_path + item,sep=',', skiprows=1)

        for header in ts_raw.columns:
            res = re.sub("[\(\[].*?[\)\]]", "", header)
            res = re.sub(r'\W+', '', res)
            ts_raw = ts_raw.rename(columns = {header:res})
        ts_data = ts_raw[['DateTimeGMT0500','TempF','Rainfallinch']]
        ts_data = ts_data.rename(columns = {'DateTimeGMT0500':'date_time',
                                          'TempF':'temp_f',
                                          'Rainfallinch': 'rain_in'})

        #Note: summer_20160802.csv is MDY, others are YMD (looks like it got opened in excel and then saved)
        if(item != '20160802.csv'):
            ts_data.date_time = pd.to_datetime(ts_data.date_time,yearfirst=True)
        else: ts_data.date_time = pd.to_datetime(ts_data.date_time)

        ts_main = ts_main.append(ts_data,sort = True)

    ts_proc = ts_main.dropna(thresh = 2).drop_duplicates().reset_index(drop = True)     # Get rows with data (all-NAN rows result from data download events)
    ts_proc['rain_in'] = ts_proc.rain_in.fillna(method = 'ffill')                       # Fill nan gaps
    #Get increments (default readout is cumulative), correcting for floating point errors
    ts_proc['incr_in'] = (ts_proc.rain_in - ts_proc.rain_in.shift()).fillna(method = 'bfill').round(decimals = 2).clip(lower = 0,upper = 0.01)
    ts_proc['qc'] = 1   # qc = 0, totally down
                        # qc = -1, clogged, but temp good
    
    for dates in down_data:
        ts_proc.loc[(ts_proc.date_time > dates[0])&(ts_proc.date_time < dates[1]),['rain_in','temp_f','incr_in','qc']] = np.nan,np.nan,np.nan,0
    for dates in clog_data:
        ts_proc.loc[(ts_proc.date_time > dates[0])&(ts_proc.date_time < dates[1]),['rain_in','incr_in','qc']] = np.nan,np.nan,-1

#    ts_proc['date_time'] = ts_proc['date_time'].dt.round('30min') #Rounds for clock lag (should never be more than a few minutes)
#    grouped = ts_proc.groupby(['date_time'])
#    result = grouped.agg(max)
#    ts_proc = result.reset_index()
    
    return ts_proc

#For NOAA data (Midway, Crete)
station_files_NOAA = ["USC00111577_1.hly","USC00112011_1.hly"]

#Column widths from NOAA-provided readme file.
w1 = [11,4,2,2,4] #Station, year, month, day, data type
w2 = [5,1,1,1,1]  #Value and four qc flags
for _ in xrange(24): w1 = w1+w2
na = [-9999,9999,999]

def stations_pull_NOAA(sta):
    df_raw = pd.read_fwf(NOAA_path+sta,    #Read fixed-width file
                         header=None, widths=w1,na_values = na)

    df_num = df_raw.select_dtypes([np.number])  #Get only columns with numerical values

    #Rename columns to something real (one column for each hour of the day)
    df_num.columns = ['year','month','day'] + [str(x) for x in np.arange(24)]

    #Melt; make the wide table long and make the hours a column themselves
    df_melty = pd.melt(df_num,df_num.columns[0:3],
                       df_num.columns[3:27],
                       var_name = 'hour')

    #Build a datetime from the pieces
    df_melty['date_time'] = pd.to_datetime(
                            df_melty.year.astype(str) + '-' +
                            df_melty.month.astype(str) + '-' +
                            df_melty.day.astype(str) + ' ' +
                            df_melty.hour + ':00:00')

    #Units are 100ths of an inch, so convert to inches
    df_melty['precip_in']=df_melty.value.astype(float)/100.0

    df_hrly = df_melty[['date_time','precip_in']]
    df_hrly = df_hrly.sort_values(by = 'date_time').reset_index(drop = True)
    return df_hrly

# For NWIS data
# '51185_00045':
# time-series 51185, unique to the Midlothian gauge
# parameter 00045 indicates that it is precipitation, inches

station_files_NWIS = ['NWIS Precip_Crete_20181109.txt','NWIS Precip_Midlothian_20181109.txt']

def stations_pull_NWIS(sta,incl_prov = False):
    param_code = '00045'
    df_NWIS = pd.read_table(NWIS_path + sta ,comment = '#',header = 0,skiprows = 1,dtype=object).drop(0)
    precip_col = [col for col in df_NWIS.columns if param_code in col and 'cd' not in col][0]
    df_NWIS['date_time'] = pd.to_datetime(df_NWIS.datetime)
    df_NWIS['precip_in'] = pd.to_numeric(df_NWIS[precip_col])
    
    if incl_prov == False: df_NWIS = df_NWIS[~df_NWIS[precip_col+'_cd'].str.contains("P")]
    df_NWIS = df_NWIS[['date_time','precip_in']]
    
    return df_NWIS


          
df_hrly_0 = stations_pull_NOAA(station_files_NOAA[0]) #Midway
df_hrly_1 = stations_pull_NOAA(station_files_NOAA[1]) #Crete
df_hrly_2 = stations_pull_NWIS(station_files_NWIS[0]) #Crete
df_15min = stations_pull_NWIS(station_files_NWIS[1],True) #Midlothian

merge_hrly = df_hrly_0.merge(df_hrly_1,on='date_time').merge(df_hrly_2,on='date_time')
merge_hrly.rename(columns={'precip_in_x':'precip_in_Midway_NOAA',
                   'precip_in_y':'precip_in_Crete_NOAA',
                   'precip_in':'precip_in_Crete_NWIS'},inplace = True)
IBP_proc = rg_combine()

IBP_proc2 = IBP_proc.set_index(pd.DatetimeIndex(IBP_proc['date_time'])).drop(['date_time'],axis = 1)

r_s15 = IBP_proc2.incr_in.resample('15Min').apply(lambda x : x.values.sum()).ffill()
t_s15 = IBP_proc2.temp_f.resample('15Min').mean().interpolate('time')
q_s15 = IBP_proc2.qc.resample('15Min').ffill()
IBP_r15 = pd.concat([r_s15,t_s15,q_s15],axis = 1).reset_index()

NWIS_s = df_15min.set_index(pd.DatetimeIndex(df_15min['date_time'])).drop(['date_time'],axis = 1).precip_in
merged_r15 = pd.merge(IBP_r15, pd.DataFrame(NWIS_s).reset_index(),how = 'left',on='date_time').sort_values(by='date_time')
#Todo: Use the QC column in IBP_r15 to replace values with those from NWIS_s
merged_r15.loc[merged_r15.qc < 1,'incr_in'] = merged_r15.loc[merged_r15.qc < 1,'precip_in']
merged_r15.drop(['precip_in'], axis = 1, inplace=True)

IBP_r30 = IBP_proc.set_index('date_time').incr_in.resample("30Min").sum()
IBP_t30 = IBP_proc.set_index('date_time').temp_f.resample("30Min").mean()
IBP_rt = pd.concat([IBP_r30,IBP_t30],axis = 1,sort = True).reset_index()
NOAA_C = merge_hrly.set_index('date_time').precip_in_Crete_NOAA
NOAA_M = merge_hrly.set_index('date_time').precip_in_Midway_NOAA
NOAA_res = pd.concat([NOAA_M,NOAA_C],axis = 1,sort=True).reset_index()
IBP_resH = IBP_proc.set_index('date_time').incr_in.resample("H").sum()
#This is short (< 1 year) because it relies on the length of the IBP time series
all_resH = pd.merge(NOAA_res,pd.DataFrame(IBP_resH.reset_index()), how = 'outer',on = 'date_time').rename(columns = {'incr_in' : 'precip_in_IBP'})
IBP_fillH = all_resH[['date_time']].copy()
IBP_fillH['flag'] = (-all_resH[['precip_in_IBP']].isnull()).astype(int)
IBP_fillH['precip_in'] = all_resH.precip_in_IBP.fillna(all_resH[['precip_in_Midway_NOAA','precip_in_Crete_NOAA']].mean(axis = 1))

merged_r15.to_csv(processed_data_path + "PRECIP_TEMP_IBP_filled_15min.csv")
IBP_rt.to_csv(processed_data_path + "PRECIP_TEMP_IBP_raw_30min.csv")
IBP_fillH.to_csv(processed_data_path + "PRECIP_IBP_filled_hrly.csv")
merge_hrly.to_csv(processed_data_path + "PRECIP_NOAA_NWIS_hrly.csv")
df_15min.to_csv(processed_data_path+"PRECIP_NWIS_15min.csv")
