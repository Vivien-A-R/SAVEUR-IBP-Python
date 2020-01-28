# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:13:21 2018

@author: Vivien

IDF curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt

rain_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Precipitation\\'

#fmin = True
fmin = False
if(fmin == True):
    fn = "PRECIP_NWIS_15min.csv"
    cn = "precip_in"
else:
    fn = "PRECIP_NOAA_hrly.csv"
    cn = "precip_in_Midway"

df_raw = pd.read_csv(rain_path + fn,index_col = 0,parse_dates = ['date_time'])
df_raw.dropna()

def K_calc(T):
    return -np.sqrt(6)/np.pi*(0.5772+np.log(np.log(T/(T-1.0))))

def f_Gumbel(x,u,B):
    return 1/B * np.exp((x-u)/B) * np.exp(-np.exp((x-u)/B))

def f_power(t,a,c,n):
    return a / (t**n + c)

def f_lognormal(z,u,s):
    return np.exp(u +s*z)

step = (len(df_raw) - 1) / 1
rain_startind = 0
while rain_startind + step < len(df_raw):
    rain_endind = rain_startind + step
    #print rain_startind,rain_endind
    df_rain = df_raw.loc[rain_startind:rain_endind].copy()
    rain_startind = rain_startind + step
    
    df_rain['raining'] = df_rain[cn] > 0
    df_rain['toggle'] = df_rain['raining'] - df_rain['raining'].shift()
    #Toggle:    +1 -> Start rain
    #           -1 -> End rain
    
    #df_rain.plot()
    rainstarts = df_rain[df_rain['toggle'] == 1].index #Padding; a gap in the rain of one hour will count as in one rainstorm
    rainends = df_rain[df_rain['toggle'] == -1].index - 1
    df_rain = df_rain[['date_time',cn]]
    
    bookends = list(map(list, zip(rainstarts, rainends)))
    
    N = len(bookends)
    dels = []
    for i in np.arange(N-1)+1:
        if(bookends[i][0] <= bookends[i-1][1]):
            bookends[i-1][1] = bookends[i][1]
            dels = dels + [i]
    for index in sorted(dels,reverse = True):
        bookends.pop(index)
    
    rows = []
    N = len(bookends)
    for i in np.arange(N-1):
        l = bookends[i][0]
        r = bookends[i][1]
        dur = (df_rain.date_time.loc[r] - df_rain.date_time.loc[l])
        hrs = dur.days*24 + dur.seconds/3600.
        
        quant = df_rain[cn].loc[l:r].sum()
        row = [df_rain.date_time.loc[l], hrs, quant]
        rows = rows + [row]
    
    df_DV = pd.DataFrame(rows)
    df_DV.columns = ['start_date','duration_hr','volume_in']
    df_DV['year'] = df_DV['start_date'].dt.year
    tabular_DV = df_DV.drop(['start_date'],axis = 1).groupby(['year','duration_hr']).max().unstack()
    
    T_desired = [2,5,10,25,50,100]
    K_det = [K_calc(t) for t in T_desired]
    Xts = []
    
    durs = np.unique(df_DV.duration_hr)
    for d in durs:
        tDV = tabular_DV['volume_in',d].sort_values(ascending = False).reset_index().reset_index().copy()
        tDV.columns = ['rank','year','volume_in']
        tDV['p'] = (tDV['rank']+1.0)/(len(tDV)+1.0)
        tDV['T'] = 1/tDV['p']
        tDV['I'] = tDV['volume_in']/d
        smean = tDV['I'].mean()
        stdev = tDV['I'].std()
        Xt = [smean + k * stdev for k in K_det]
        Xts = Xts + [Xt]
    
    df_IDF = pd.DataFrame(Xts)
    df_IDF.columns = T_desired
    df_IDF.index = durs
    
    df_IDF.dropna(thresh = 2).plot(use_index = True)
    
    fdurs = list(np.linspace(0.25,16,64))
    fits = []
        
    for c in df_IDF.columns:
        test = df_IDF[c].dropna()
        fitparams = scopt.curve_fit(f_lognormal,test.index,test)
        ufit,sfit = fitparams[0][0],fitparams[0][1]
        test = test.reset_index()
        fit = [f_lognormal(d,ufit,sfit) for d in fdurs]
        fits = fits + [fit]
        
    df_IDF_fit = pd.DataFrame(fits,columns = fdurs, index = T_desired).transpose()
    ax = df_IDF_fit.plot(use_index = True, xlim = [0,10],ylim = [0,2.5])
    ax.set_xlabel("Duration (hr)")
    ax.set_ylabel("Intensity (in/hr)")
    ax.legend(title = "Frequency (return period, yr)")

df_DI = df_DV.drop(['year'],axis = 1)
df_DI = df_DI[df_DI['duration_hr'] < 24]
df_DI['intensity_in_hr'] = df_DI['volume_in']/df_DI['duration_hr']
df_DI = df_DI[df_DI['intensity_in_hr'] < 2]
df_DI = df_DI[df_DI['intensity_in_hr'] != 0.1]
df_DI = df_DI[df_DI['intensity_in_hr'] != 0.01]

## Calculate CDFs for precip.
plt.figure()
hist,bins = np.histogram(np.array(df_DI.intensity_in_hr),bins = 50)
plt.bar(bins[1:],hist,width = (bins[1]-bins[0])/2)
plt.title("Intensity (in/hr)")

plt.figure()
hist,bins = np.histogram(np.array(df_DI.volume_in),bins = 25)
plt.bar(bins[1:],hist,width = (bins[1]-bins[0])/2)
plt.title("Event Volume (in)")

plt.figure()
hist,bins = np.histogram(np.array(df_DI.duration_hr),bins = 10)
plt.bar(bins[1:],hist,width = (bins[1]-bins[0])/2)
plt.title("Duration (hr)")

df_DI.plot(x = 'start_date',subplots = True, sharex = True,kind = 'bar')

fig, [ax1, ax2, ax3] = plt.subplots(3,1,sharex = True, sharey = False)
ax1.plot(df_DI.start_date,df_DI.duration_hr,'.',label = "Duration (hr)")
ax2.plot(df_DI.start_date,df_DI.volume_in,'.',label = "Volume (in)")
ax3.plot(df_DI.start_date,df_DI.intensity_in_hr,'.',label = "Intensity (in/hr)")
ax1.legend()
ax2.legend()
ax3.legend()
fig.autofmt_xdate()

arr_precip_sorted = np.array(df_DI.intensity_in_hr.sort_values(ascending = True))
pdf_precip = arr_precip_sorted/arr_precip_sorted.sum()
cdf_precip = pdf_precip.cumsum()
plt.figure()
plt.plot(arr_precip_sorted,cdf_precip,'.') #This is it!
plt.title("cdf of intensity")
