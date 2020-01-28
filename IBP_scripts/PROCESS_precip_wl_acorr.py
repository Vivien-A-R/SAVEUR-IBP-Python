# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:58:01 2018

@author: Vivien
"""

import pandas as pd
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

#Correlation analysis
well_path = 'C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Water Level Derived Products\FilledSmoothed\\'
meta_path = 'C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'
rain_path = 'C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Precipitation\\'
soil_path = 'C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Data From SMP\\'

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    
    https://stackoverflow.com/questions/33171413/cross-correlation-time-lag-correlation-with-pandas
    """
    return datax.corr(datay.shift(lag))

def days(td):
    return td.days+td.seconds/3600./24

def days_hours_minutes(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60

xy_pos = pd.read_csv("voronoi_areas.csv")
vor_areas = xy_pos[['sensor','encl_area_m']].copy()
#xy_pos = pd.read_csv("voronoi_areas_2016-2017.csv")
#vor_areas_OLD = xy_pos[['sensor','encl_area_m']].copy()

## IBP Precip Time series, hourly; using cumsum for dz effectively downsamples
df_rain = pd.read_csv(rain_path + "PRECIP_IBP_filled_hrly.csv",parse_dates = ['date_time'],index_col = 0).fillna(0)
df_rain['incr_m']  = df_rain['precip_in']*0.0254
df_rain['cum_m'] = df_rain['precip_in'].cumsum()*0.0254
df_rain['cum_detrend'] = sps.detrend(df_rain.cum_m)
df_rain['rain_dV'] = df_rain.incr_m*xy_pos.encl_area_m.sum()

sensor_meta = pd.read_table(meta_path+'wl_position_meta.csv',sep=',',index_col = None)

ts_dV = pd.DataFrame({'date_time':[]}) # This is a dataframe of volumes that will contain all of the wells, arranged by datetime. Then we can look at dV prairie-wide and plot spatially!
wells = ['WLW1','WLW2','WLW3','WLW4','WLW5','WLW6','WLW7','WLW8','WLW9','WLW10','WLW12','WLW13','WLW14']
#wells = ['WLW2']
well_id = 'WLW2'
porosities = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
porosities = [0.5]
phi = 0.5

#dp_suff = "_f5_s16.csv"
#dp_suff = "_f5_s8.csv"
dp_suff = "_f5_lowess.csv"

for well_id in wells:
#for phi in porosities:
    print("\n" + well_id + "\nLocal, not detrended, " + str(int(phi*100)) + "% porosity")
    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
    f.suptitle(well_id)
    gelev_m = sensor_meta[sensor_meta.sensor == well_id].ground_elev_ft.drop_duplicates().item() * 0.3048
    df_well = pd.read_csv(well_path + well_id + dp_suff, parse_dates = ['date_time'], index_col = None)
    
    #WL above ground surface (meter)
    df_well['wl_m_ags'] = df_well['sensor_elev_m'] + df_well['depth_m_smoothed'] - gelev_m
    df_well.loc[df_well.qual_c < 1,'wl_m_ags'] = np.nan
    df_well = df_well[df_well.wl_m_ags.first_valid_index():df_well.wl_m_ags.last_valid_index()+1].reset_index(drop = True)
    df_well['wl_detrend']  = sps.detrend(df_well.wl_m_ags.interpolate('nearest'))
    df_well['dz'] = df_well['wl_m_ags'].interpolate('nearest').diff()
    df_well.loc[df_well.dz < 0,'dz'] = 0        # Get only the increases
    df_well.loc[df_well.wl_m_ags < 0, 'dz'] = df_well.loc[df_well.wl_m_ags < 0, 'dz'] * phi # Account for porosity
    df_well['cum_dz'] = df_well.dz.cumsum()
    df_well['dz_detrend'] = sps.detrend(df_well.cum_dz.bfill())
    df_well['dV'] = df_well['dz']*xy_pos[xy_pos['sensor'] == well_id].encl_area_m.item()
    
    ts_dV = ts_dV.merge(df_well[['date_time','dV']].rename(columns = {'dV':'dV_'+well_id}),how = 'outer',on = 'date_time',sort = True)
    
    df_combine = pd.merge(df_rain[['date_time','incr_m','cum_m','cum_detrend']], 
                          df_well[['date_time','wl_m_ags','wl_detrend','dz','cum_dz','dz_detrend']], 
                          on ='date_time',how = 'inner')
    df_combine.columns = ['date_time','precip_incr','precip_cum','precip_cum_det','wl_ags','wl_det','wl_dz_pos','wl_dz_pos_cum','wl_dz_pos_cum_det']
    df_combine.precip_cum = df_combine.precip_cum - df_combine.precip_cum[0].item()
    df_combine.plot(x='date_time', 
                    y = ['precip_incr','wl_dz_pos'], 
                    ax = ax1)
    
    factor = df_combine.wl_dz_pos_cum.iloc[-1]/df_combine.precip_cum.iloc[-1]
    factor = 1
    df_combine['wl_dz_pos_cum_corr'] = df_combine['wl_dz_pos_cum']/factor
    df_combine.plot(x = ['date_time'], y = ['precip_cum','wl_dz_pos_cum_corr'],ax=ax2)
    df_combine.plot(x = ['precip_cum'], y = ['wl_dz_pos_cum_corr'],ax=ax3)
    
    hours = range(0,60*24)
    xcov_hour = [crosscorr(df_combine.precip_cum,df_combine.wl_dz_pos_cum,lag = i) for i in hours]
    ax4.plot([i/24.0 for i in hours],xcov_hour)
    print df_combine.loc[xcov_hour.index(max(xcov_hour))].date_time - df_combine.date_time.loc[0]

ts_dV_reduced = ts_dV.dropna()
ts_dV_reduced['well_sum'] = ts_dV.sum(axis = 1)
fig = plt.figure()
plt.plot(ts_dV_reduced.date_time,ts_dV_reduced.well_sum)
plt.plot(df_rain.date_time,df_rain.rain_dV)
plt.legend()