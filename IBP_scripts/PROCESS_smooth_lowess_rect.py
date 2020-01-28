# -*- coding: utf-8 -*-
"""
Fill gaps and smooth data using both rectangular rolling window and LOWESS

Created on Mon Aug 20 15:09:00 2018

@author: Vivien
"""

import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import groupby
import matplotlib.pyplot as plt
import statsmodels.nonparametric.smoothers_lowess as sml
from scipy.interpolate import interp1d

# Import data
root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\\'
data_path = root_path + 'Documents\Processed Water Level Data\\'
processed_path = root_path + '\Documents\Water Level Derived Products\FilledSmoothed\\'

sensor_meta = pd.read_table(data_path+'wl_position_meta.csv',sep=',',index_col = None)

# Fill gaps with average of *avrange* measurements for either side of gap
# Useful for frozen sensors or brief sensor-out-of-water moments
# Skips data missing for more than one day (no fill, leave the dry wells dry)
def fillgaps(sensor_id, avrange = 5):
    print(sensor_id)
    df_w = pd.read_table(data_path + sensor_id + '_ibp_main.csv',sep=',',parse_dates=['date_time'])
    
    print("Filling gaps")    
    replace_index = df_w[df_w.qual_c < 1].index
    ranges = []
    for k, g in groupby(enumerate(replace_index), lambda (i,x):i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    
    for l in ranges:
        span = l[1]-l[0]
        frozen = df_w.qual_c[l[0]:l[1]].mean() < -1
        if(span < 48 or frozen == True):
            ris_l = l[0] - avrange
            ris_r = l[1] + avrange
    
            if(len(df_w.depth_m.iloc[l[1] + 1:ris_r]) > 0):
                df_w.depth_m.iloc[l[0]-1:l[1]+1] = np.mean([np.mean(df_w.depth_m.iloc[ris_l:l[0]]),np.mean(df_w.depth_m.iloc[l[1]+1:ris_r])])
                df_w.WS_elevation_m.iloc[l[0]-1:l[1]+1] = np.mean([np.mean(df_w.WS_elevation_m.iloc[ris_l:l[0]]),np.mean(df_w.WS_elevation_m.iloc[l[1]+1:ris_r])])
            else:
                df_w.WS_elevation_m.iloc[l[0]] = np.mean(df_w.WS_elevation_m.iloc[ris_l:l[0]])
            
            df_w.qual_c.iloc[l[0]-1:l[1]+1] = 2
            
    return df_w
    #n_rows, n_cols = df_w.shape

# Smooth using rolling window of *window_size* measurements (consider other methods);
def smooth_rect(df_w,window_size = 16):
    print("Smoothing")
    
    # Smooth data
    hw = window_size/2
    l = len(df_w)    

    depth_mean = df_w.depth_m.rolling(window_size,
                                      center=True,
                                      win_type='boxcar').mean()     # rolling mean boxcar filter to smooth data. 
    depth_mean[0:hw] = depth_mean[hw + 1]                           # handles the beginning points where the filter doesn't apply
    depth_mean[l - hw:l] = depth_mean[l - hw]                       # handles the final points where the filter doesn't apply
    df_w['depth_m_smoothed'] = depth_mean                           # add to the dataframe.

    return df_w

def smooth_lowess(df_w):
    x = np.array(df_w['run_time'])
    y = np.array(df_w['depth_m'])
    lowess_xy = sml.lowess(y,x,0.0005,delta = 0.0001*max(x))
    [lowess_x,lowess_y] = list(zip(*lowess_xy))
    f = interp1d(lowess_x, lowess_y, bounds_error=False)
    xnew = range(int(x[0]),int(x[-1])+1800,1800)
    ynew = f(xnew)
    dfnew = pd.DataFrame(list(zip(xnew,ynew)),columns = ['run_time','depth_m_smoothed'])
    df_w = df_w.merge(dfnew,on='run_time')

    return df_w
    

#fig1, (ax1,ax2) = plt.subplots(2,1,sharex = True)
#fig2, (ax3,ax4) = plt.subplots(2,1)
#fig3, ax5 = plt.subplots()
s = ["WLW1","WLW2","WLW3","WLW4","WLW5","WLW6","WLW7","WLW8","WLW9","WLW10","WLW12","WLW13","WLW14"]

nfill = 5
nsmooth = 8
for sensor_id in s:

    df = smooth_rect(fillgaps(sensor_id,nfill),nsmooth)
    df.plot(x = 'date_time',y = ['depth_m','depth_m_smoothed'],title = sensor_id)
    namestring = processed_path + sensor_id + '_f' + str(nfill) + "_s" + str(nsmooth) + ".csv"
    df.to_csv(namestring ,index = None)
    
    df = smooth_lowess(fillgaps(sensor_id,nfill))
    df.plot(x = 'date_time',y = ['depth_m','depth_m_smoothed'],title = sensor_id)
    namestring = processed_path + sensor_id + '_f' + str(nfill) + "_lowess" + ".csv"
    df.to_csv(namestring ,index = None)
    
