# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:10:06 2017

@author: Vivien
"""

import pandas as pd
import numpy as np

#import seaborn.apionly as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.dates as mdates

import soiltype_to_numeric as sn
from cdf_violin2 import cdf

mpl.rcParams.update({'font.size': 16})

def clamp(val, minimum=0, maximum=1):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def mplcolors(ctuple):
    x = [y/255. for y in ctuple]
    return x

def colorshift(ctuple,degree):
    x = [clamp(y * degree) for y in ctuple]
    return x
    

cc=[[135,206,250],[250,128,114],[173,255,47],[160,82,45],[205,175,149],[106,90,205]]
color_column = [colorshift(mplcolors(l),0.7) for l in cc]

labels = ['CL','CH','SP','ML','OL','Sandy CL']

mpath = "C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\\"

gwpath = mpath + "IBP Project\Documents\Deprecated Data Folder\Processed Water Level Data\\"
metadata = pd.read_csv(gwpath + 'wl_position_meta.csv')
re_int = "2H"

#Example plotting with a chosen time series
well_ch = ["WLW3","WLW4"]
surf_ch = ["WLS3"]

gw0 = pd.read_csv(gwpath + well_ch[0] + "_ibp_main.csv",parse_dates = ['date_time'])
gw0.loc[gw0.qual_c < 1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
gw0 = gw0[['date_time','WS_elevation_m']]
gw0_res= gw0.set_index('date_time').WS_elevation_m.resample(re_int).mean().to_frame().reset_index()

gw1 = pd.read_csv(gwpath + well_ch[1] + "_ibp_main.csv",parse_dates = ['date_time'])
gw1.loc[gw1.qual_c < 1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
gw1 = gw1[['date_time','WS_elevation_m']]
gw1_res= gw1.set_index('date_time').WS_elevation_m.resample(re_int).mean().to_frame().reset_index()

sw0 = pd.read_csv(gwpath + surf_ch[0] + "_ibp_main.csv",parse_dates = ['date_time'])
sw0.loc[sw0.qual_c < 1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
sw0 = sw0[['date_time','WS_elevation_m']]
sw0_res= sw0.set_index('date_time').WS_elevation_m.resample(re_int).mean().to_frame().reset_index()

ws = pd.merge(gw0,gw1, suffixes = well_ch,on='date_time')
ws = pd.merge(ws,sw0,on='date_time')
ws = ws.rename(columns = {"WS_elevation_m":"WS_elevation_mWLS3"})

smpath = mpath + "IBP Project\Documents\Deprecated Data Folder\Data From SMP\\"
sm = pd.read_csv(smpath + 'SMP2_ibp_main.csv',parse_dates = ['date_time'])
sm_res1 = sm.set_index('date_time').a1_moisture.resample(re_int).mean().to_frame().reset_index()
sm_res2 = sm.set_index('date_time').a2_moisture.resample(re_int).mean().to_frame().reset_index()
sm_res3 = sm.set_index('date_time').a3_moisture.resample(re_int).mean().to_frame().reset_index()
sm_res4 = sm.set_index('date_time').a4_moisture.resample(re_int).mean().to_frame().reset_index()
sm_res5 = sm.set_index('date_time').a5_moisture.resample(re_int).mean().to_frame().reset_index()
sm_res6 = sm.set_index('date_time').a6_moisture.resample(re_int).mean().to_frame().reset_index()


rpath = mpath + "IBP Project\Documents\Deprecated Data Folder\Precipitation\\"
rp = pd.read_csv(rpath + 'PRECIP_ibp_main.csv',parse_dates = ['date_time'])
rp['precip_cm'] = rp['precip_in']*2.54
rp = rp.resample(re_int,on='date_time').sum().reset_index()
rp_IBP = rp.copy()
rp_IBP.loc[rp_IBP.flag == 0,"precip_cm"] = np.nan
rp_CW = rp.copy()
rp_CW.loc[rp_CW.flag > 0,"precip_cm"] = np.nan

merged = pd.merge(rp_IBP[['date_time','precip_cm']],rp_CW[['date_time','precip_cm']],on = 'date_time',suffixes = ['_IBP','_CW'])
merged = pd.merge(merged, ws)
merged = pd.merge(merged, sm_res1)
merged = pd.merge(merged, sm_res2)
merged = pd.merge(merged, sm_res3)
merged = pd.merge(merged, sm_res4)
merged = pd.merge(merged, sm_res5)
merged = pd.merge(merged, sm_res6)

merged.rename(columns = {'a1_moisture':'VWC%_10cm',
                         'a2_moisture':'VWC%_20cm',
                         'a3_moisture':'VWC%_40cm',
                         'a4_moisture':'VWC%_60cm',
                         'a5_moisture':'VWC%_80cm',
                         'a6_moisture':'VWC%_100cm'},inplace = True)

df = merged.copy()
timecrop = ['2016-07-08','2016-07-30']
df = df[(df['date_time'] > timecrop[0]) & (df['date_time'] < timecrop[1])]
merged = df.copy()    
    
def wlplot():
    global merged
    df = merged.reset_index().copy()
    timecrop = ['2016-07-08','2016-07-30']
    df = df[(df['date_time'] > timecrop[0]) & (df['date_time'] < timecrop[1])]
    merged = df.copy()
    f, (ax2) = plt.subplots(1, sharex=True,figsize = [1,1])
    #SM
    lw = 2 #linewidth
    lp = 1.5 #label padding
    lp2 = 0.5 #ticklabel padding

    
    #Raw rain signal
    ax3=ax2.twinx()
    rain, = ax3.step(merged.date_time,merged.precip_cm_IBP,color = 'black')
    ax3.step(merged.date_time,merged.precip_cm_CW,color = 'darkgray',zorder = 1)
    ax3.set_ylim([0,3]) 
    ax3.set_ylabel("Rain (cm)",color = 'black')
    ax3.yaxis.labelpad=lp
    ax3.tick_params(axis='y', direction = 'in',pad = lp2)
    
    
    #Raw groundwater signal
    water, = ax2.plot(merged['date_time'],merged['WS_elevation_mWLW3']-184,color = 'blue',zorder = 2)
    #ax2.plot(merged['date_time'],merged['WS_elevation_mWLW4']-184,color = 'blue')
    #ax2.legend([water, rain], ["Groundwater", "Precip."],loc = 4,frameon = True)
    ax2.set_ylabel("Water\nElev. (m)",color = 'blue')
    ax2.yaxis.labelpad=lp
    ax2.set_ylim([-0.2,0.8])
    ax2.xaxis.labelpad=lp
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.tick_params(axis='both', direction = 'in',pad = lp2,color = 'blue',labelcolor= 'blue')
    ax2.xaxis.set_ticklabels([])
    
    
    f.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    f.autofmt_xdate()
    
def splot():
    global merged
    f, (ax4,ax2,ax1) = plt.subplots(3, sharex=True,figsize = [12,12],)
    #SM
    lw = 2 #linewidth
    lp = 2 #label padding
    lp2 = 0.5 #ticklabel padding
    ax1.plot(merged['date_time'],merged['VWC%_10cm'],color = 'black',linewidth = lw,label = '10cm')
    #ax1.plot(merged['date_time'],merged['VWC%_10cm'],color = color_column[0],dashes=[6,1],linewidth = lw,label = '10cm')
    #ax1.plot(merged['date_time'],merged['VWC%_20cm'],color = color_column[0],dashes=[8,1],linewidth = lw,label = '20cm')
    #ax1.plot(merged['date_time'],merged['VWC%_40cm'],color = color_column[0],dashes=[12,1],linewidth = lw,label = '40cm')
    #ax1.plot(merged['date_time'],merged['VWC%_60cm'],color = color_column[0],dashes=[6,1],linewidth = lw,label = '60cm')
    #ax1.plot(merged['date_time'],merged['VWC%_80cm'],color = color_column[1],dashes=[8,1],linewidth = lw,label = '80cm')
    #ax1.plot(merged['date_time'],merged['VWC%_100cm'],color = color_column[1],dashes=[12,1],linewidth = lw,label = '100cm')
    #ax1.legend(handlelength = 3,loc = 4)
    ax1.tick_params(axis='both', direction = 'in',pad = lp2)
    ax1.set_ylabel("Soil Moisture\n(%VWC)")
    ax1.yaxis.labelpad=lp
    
    #Raw rain signal
    ax3=ax4.twinx()
    #rain, = ax3.step(merged.date_time,merged.precip_cm_IBP,color = 'black')
    merged  = merged.set_index('date_time')
    rain = ax3.fill_between(merged.index,merged.precip_cm_IBP,0,color = 'black',step = 'post')
    ax3.step(merged.index,merged.precip_cm_CW,color = 'darkgray')
    ax3.set_ylim([3,0]) 
    ax3.set_ylabel("Rain (cm)",color = 'black')
    ax3.yaxis.labelpad=lp
    ax3.tick_params(axis='y', direction = 'in',pad = lp2)
    
    
    #Raw groundwater signal
    water, = ax2.plot(merged['WS_elevation_mWLW3'],color = 'blue')
    ax2.plot(merged['WS_elevation_mWLW4'],color = 'blue')
    #ax2.legend([water, rain], ["Groundwater", "Precip."],loc = 4,frameon = True)
    ax2.set_ylabel("Water\nElev. (m)",color = 'blue')
    ax2.set_ylim([184,184.9]) 
    ax2.yaxis.labelpad=lp
    ax2.xaxis.labelpad=lp
    #ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.tick_params(axis='y', direction = 'in',pad = lp2,color = 'blue',labelcolor= 'blue')
    #ax2.xaxis.set_ticklabels([])
    
    #Raw surface signal
    water, = ax4.plot(merged['WS_elevation_mWLS3'],color = 'blue')
    #ax2.legend([water, rain], ["Groundwater", "Precip."],loc = 4,frameon = True)
    ax4.set_ylabel("Water\nElev. (m)",color = 'blue')
    ax4.set_ylim([184,184.9])
    ax4.yaxis.labelpad=lp
    ax4.xaxis.labelpad=lp
    ax4.tick_params(axis='y', direction = 'in',pad = lp2,color = 'blue',labelcolor= 'blue')
    #ax2.xaxis.set_ticklabels([])
    
    f.subplots_adjust(hspace=0)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    #f.autofmt_xdate()
    myFmt = mdates.DateFormatter('%B %d')
    ax1.xaxis.set_major_formatter(myFmt)