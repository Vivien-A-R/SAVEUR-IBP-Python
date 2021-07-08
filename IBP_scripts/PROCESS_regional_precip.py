# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:28:04 2021

@author: Packman-Field
"""

import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import matplotlib as mpl
from meta_get import meta_get

raw_path = "C:\Users\Packman-Field\Documents\Paper II\Water Data\\"

def stations_pull_NOAA():
    df_raw = pd.read_csv(raw_path + "\Crete_HRLY.csv",low_memory = False)

    df_num = df_raw.select_dtypes([np.number]).iloc[:,3:-1]  #Get only columns with numerical values

    #Rename columns to something real (one column for each hour of the day)
    df_num.columns = [str(x) for x in np.arange(24)]
    df_num['date'] = df_raw['DATE']

    #Melt; make the wide table long and make the hours a column themselves
    df_melty = pd.melt(df_num,df_num.columns[-1],
                       df_num.columns[:-1],
                       var_name = 'hour')

    #Build a datetime from the pieces
    df_melty['date_time'] = pd.to_datetime(
                            df_melty.date.astype(str) + " " + 
                            df_melty.hour.astype(str) + ':00:00')
    
    df_melty.value.replace(-9999,np.nan,inplace = True)
    
    #Units are 100ths of an inch, so convert to inches
    df_melty['precip_in']=df_melty.value.astype(float)/100.0

    df_hrly = df_melty[['date_time','precip_in']]
    df_hrly = df_hrly.sort_values(by = 'date_time').reset_index(drop = True)
    
    return df_hrly

def stations_pull_USGS():
    df_raw = pd.read_csv(raw_path + "\Crete_15min.txt",sep='\t',    #Read fixed-width file
                         skiprows = 29,
                         header = None)
    df_raw.columns = ['origin','station','date_time','tz','precip_in','qc']
    df_raw.date_time = pd.to_datetime(df_raw.date_time)
    
    df_irr = df_raw[['date_time','precip_in']].copy()
    return df_irr

crete_irr = stations_pull_USGS()
crete_hrly = stations_pull_NOAA()

#pd.plot(crete_irr)


crete_all = pd.merge(crete_irr,crete_hrly,how='outer',on = 'date_time')
crete_all['precip_in']=crete_all[['precip_in_x','precip_in_y']].mean(axis = 1) #combined USGS and NOAA data

crete_avg = crete_all[['date_time','precip_in']].copy().sort_values("date_time", axis = 0) 
crete_avg.set_index('date_time', inplace = True)

crete_avg.to_csv(raw_path+"precip_in_Crete.csv")