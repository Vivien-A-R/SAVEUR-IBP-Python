# -*- coding: utf-8 -*-
"""
Created on Tue May 08 11:27:58 2018

@author: Vivien
"""

import matplotlib.pyplot as plt
import pandas as pd
from numpy import nan

root_path  = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\'
processed_data_path = root_path + 'Data from Em50s\\'
rain_path = root_path + 'Precipitation\\'
WLS_path = root_path + 'Processed Water Level Data\\'

xlims = ['2017-10-01 00:00:00','2018-10-01 00:00:00']

def ec_plot1(sensors = ['ECL1','ECL2','ECL3','ECL4','ECL5','ECL6']):
    for id in sensors:
        df = pd.read_csv(processed_data_path + id + '_ibp_main.csv',parse_dates = ['date_time'])

        f, (ax1,ax2,ax3) = plt.subplots(3, sharex=True,figsize = [6,4],gridspec_kw = {'height_ratios':[1,1,1]})
        ax1.plot(df.date_time, df.A_VWC_frac)
        ax1.plot(df.date_time, df.B_VWC_frac)
        ax2.plot(df.date_time, df['A_EC_mS/cm'])
        ax2.plot(df.date_time, df['B_EC_mS/cm'])
        ax3.plot(df.date_time, df.A_Temp_C)
        ax3.plot(df.date_time, df.B_Temp_C)
        if (len(df.columns) == 9):
            #ax2.plot(df.date_time, df['w_EC_mS/cm'])
            ax3.plot(df.date_time, df.w_Temp_C)
        
        ax1.set_xlim(xlims)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        
        f.suptitle(id)

df_rain = pd.read_csv(rain_path + "PRECIP_TEMP_IBP_raw_30min.csv",index_col = 0,parse_dates=['date_time'])
df_rain['temp_c'] = (df_rain['temp_f']-32.) * 5./9.
df_rain['incr_cm'] = df_rain['incr_in'] * 2.54
df_rain.drop(['temp_f','incr_in'],axis = 1, inplace=True)

        
set1 = ['ECL1','ECL2']                  # South ditch
set2 = ['ECL3','ECL4','ECL5','ECL6']    # North ditch

ncolors = ['blue','red','green','purple','orange','magenta']
dcolors = ['midnightblue','darkred','darkgreen','indigo','chocolate','darkmagenta']

#Plot EC, temperature, and precip + one water level surface sensor
def ec_plot2(sensors = set2):
    no_WECS = True
    if any(s in sensors for s in set1):
        wls_id = "WLS5"
        wls_el = (613.705 - 608.234 - 5.33)*0.3048
    if any(s in sensors for s in set2):
        wls_id = "WLS8"
        wls_el = (608.080 - 602.676 - 4.81)*0.3048
    
    df_WL = pd.read_csv(WLS_path + wls_id + "_ibp_main.csv",parse_dates = ['date_time'])
    df_WL.loc[df_WL.qual_c < 1,"depth_m"]=nan #Skip already-flagged values
    
    for id in sensors:
        df = pd.read_csv(processed_data_path + id + '_ibp_main.csv')
        if (len(df.columns)==9): no_WECS = False
    
    if no_WECS == False:        
        f, (ax2,ax2b,ax3,ax4) = plt.subplots(4, sharex=True,figsize = [8,10],gridspec_kw = {'height_ratios':[3,3,2,3]})
        ax4b = ax4.twinx()
        ax2b.set(ylabel = 'Water EC, mS/cm')
    else:
        f, (ax2,ax3,ax4) = plt.subplots(3, sharex=True,figsize = [8,10],gridspec_kw = {'height_ratios':[3,2,3]})
        ax4b = ax4.twinx()
    
    cc = 0
    for id in sensors:
        df = pd.read_csv(processed_data_path + id + '_ibp_main.csv',parse_dates = ['date_time'])
        ax2.plot(df.date_time, df['A_EC_mS/cm'],color = ncolors[cc])
        ax2.plot(df.date_time, df['B_EC_mS/cm'],color = ncolors[cc],linestyle=':')
        
        if (len(df.columns)==9):
            ax2b.plot(df.date_time, df['w_EC_mS/cm'],color = 'black')
            #ax3.plot(df.date_time, df.w_Temp_C,color = 'black')
        cc = cc + 1
    
    ax3.plot(df_rain.date_time,df_rain.temp_c,color = 'black')
    ax3.axhline(y=0,color = 'gray',linestyle = '--')
    ax4.plot(df_rain.date_time,df_rain.incr_cm,color = 'black')
    ax4b.plot(df_WL.date_time,(df_WL.depth_m + wls_el)*100,color = 'teal')
        
    ax2.set(ylabel = 'Soil EC, mS/cm',ylim = [-0.05,0.5],xlim = xlims)
    ax3.set(ylabel = 'Temp., C')
    ax4.set(ylabel = 'Precip., cm',ylim = [0,2.05])
    ax4b.set(xlabel = 'time (minutes)', ylabel = 'Water depth, cm',ylim = [0,150])
    
    #ax2.legend()
    #ax3.legend()
    
    f.suptitle(sensors)
    f.subplots_adjust(hspace=0)
    f.autofmt_xdate()

#Plot all of the above + soil moisture!    
def ec_plot3(sensors = set2):
    no_WECS = True
    
    if any(s in sensors for s in set1):
        wls_id = "WLS5"
        wls_el = (613.705 - 608.234 - 5.33)*0.3048
    if any(s in sensors for s in set2):
        wls_id = "WLS8"
        wls_el = (608.080 - 602.676 - 4.81)*0.3048
    
    df_WL = pd.read_csv(WLS_path + wls_id + "_ibp_main.csv",parse_dates = ['date_time'])
    df_WL.loc[df_WL.qual_c < 1,"depth_m"]=nan #Skip already-flagged values
    
    for id in sensors:
        df = pd.read_csv(processed_data_path + id + '_ibp_main.csv')
        if (len(df.columns)==9): no_WECS = False
    
    if no_WECS == False:        
        f, (ax1,ax2, ax2b, ax3, ax4) = plt.subplots(5, sharex=True,figsize = [8,10],gridspec_kw = {'height_ratios':[3,3,3,2,3]})
        ax4b = ax4.twinx()
        ax2b.set(ylabel = 'Water EC, mS/cm')
    else:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True,figsize = [8,10],gridspec_kw = {'height_ratios':[3,3,2,3]})
        ax4b = ax4.twinx()
    
    cc = 0
    for id in sensors:
        df = pd.read_csv(processed_data_path + id + '_ibp_main.csv',parse_dates = ['date_time'])
        ax1.plot(df.date_time, df['A_VWC_frac'],color = ncolors[cc])
        #ax1.plot(df.date_time, df['B_VWC_frac'],color = ncolors[cc],linestyle=':')        
        ax2.plot(df.date_time, df['A_EC_mS/cm'],color = ncolors[cc])
        #ax2.plot(df.date_time, df['B_EC_mS/cm'],color = ncolors[cc],linestyle=':')
        
        if (len(df.columns)==9):
            ax2b.plot(df.date_time, df['w_EC_mS/cm'],color = 'black')
            #ax3.plot(df.date_time, df.w_Temp_C,color = 'black')
        cc = cc + 1
    
    ax3.plot(df_rain.date_time,df_rain.temp_c,color = 'black')
    ax3.axhline(y=0,color = 'gray',linestyle = '--')
    ax4.plot(df_rain.date_time,df_rain.incr_cm,color = 'black')
    ax4b.plot(df_WL.date_time,(df_WL.depth_m + wls_el)*100,color = 'teal')
    
    ax1.set(ylabel = 'VWC_frac')    
    ax2.set(ylabel = 'Soil EC, mS/cm',ylim = [-0.05,0.5], xlim = xlims)    
    ax3.set(ylabel = 'Temp., C')
    ax4.set(ylabel = 'Precip., cm',ylim = [0,2.05])
    ax4b.set(xlabel = 'time (minutes)',ylabel = 'Water depth, cm',ylim = [0,150])
    
    #ax2.legend()
    #ax3.legend()
    
    f.suptitle(sensors)
    f.subplots_adjust(hspace=0)
    f.autofmt_xdate()
    
ec_plot2()
ec_plot3()