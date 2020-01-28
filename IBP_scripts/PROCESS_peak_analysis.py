# -*- coding: utf-8 -*-
"""
Takes text-based data (boring logs) and assigns numeric (integer) codes to each unique value
For PCA or other numeric/categorical analysis
Created on Fri May 11 12:43:50 2018

@author: Vivien
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from operator import itemgetter
from itertools import groupby
import scipy as sc

# Import data
data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Deprecated Data Folder\Processed Water Level Data\\'
sensor_id = "WLW4"
rain_path = "C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Deprecated Data Folder\Precipitation\\"
rain = pd.read_csv(rain_path + "PRECIP_IBP_filled_hrly.csv",index_col = 0,parse_dates = ['date_time'])

#Define forms of functions for fitting purposes
def expn_func(x,a,b):
    return a**x + b

def powr_func(x,k,n,b):
    return k*x**n + b

def linr_func(x,m,b):
    return m*x+b


fig1, (ax1,ax2) = plt.subplots(2,1,sharex = True)
fig2, (ax3,ax4) = plt.subplots(2,1)
fig3, ax5 = plt.subplots()
s = ["WLW1","WLW2","WLW3","WLW4","WLW5","WLW6"]
#s = ["WLW4"]
dtable = []
for sensor_id in s:
    print(sensor_id)
    df_w=pd.read_table(data_path + sensor_id + '_ibp_main.csv',sep=',',parse_dates=['date_time'])
    
    replace_index = df_w[df_w.qual_c < 1].index
    ranges = []
    for k, g in groupby(enumerate(replace_index), lambda (i,x):i-x):
        group = map(itemgetter(1), g)
        ranges.append((group[0], group[-1]))
    
    print("Filling gaps")
    avrange = 5
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
    n_rows, n_cols = df_w.shape
    
    print("Smoothing")
    # Smooth data
    window_size = 16 #the number of data points to include in the rolling window.
    hw = window_size/2
    depth_mean = df_w.depth_m.rolling(window_size,center=True,win_type='boxcar').mean() # rolling mean boxcar filter to smooth data. 
    depth_mean[0:hw] = depth_mean[hw + 1] # handles the beginning points where the filter doesn't apply. This needs to be altered if window_size is changed
    depth_mean[len(depth_mean) - hw:len(depth_mean)] = depth_mean[len(depth_mean) - hw] # handles the final points where the filter doesn't apply
    df_w['depth_ave'] = depth_mean # add to the dataframe.

    #Get rid of extra information
    df_w.drop(['pressure_pa','temperature_c','qual_c','sensor_elev_m','WS_elevation_m'],axis = 1, inplace = True)
    
    print("Finding peaks")
    # Find peaks
    
    #def sl(oneroll):
    #    #do regression on the rolling window
    #    return slope
    #rolls = df_w[['run_time','depth_ave']].rolling(window = 8,center = True)
    #df_w['slope'] = rolls.apply(sl)
    #pandas rolling window apply does NOT work here because this version it only takes one column. Boo.
    # TODO: Update Pandas version and rewrite this neatly
    
    # Set up rolling window params
    roll_ws = 8
    hws = roll_ws/2 # half the size of the window over which the regression will work.
    
    # First derivative
    txs=np.zeros(shape=[1,n_rows]) #temporary storage variable for the slope
    txm=np.zeros(shape=[1,n_rows]) #temporary storage variable for the mean
    
    for n in range(hws, n_rows - hws): ##
        temp_regress_y = df_w.depth_ave[n-hws:n+hws] #selects data to regress, could reduce noise even further by selecting the smoothed data from above.
        temp_regress_x = df_w.run_time[n-hws:n+hws] # x-coordinates for regression
        slope, intercept, r_value, p_value, std_err = sc.stats.linregress(temp_regress_x/(60.0**2),temp_regress_y) #linear regression
        txs[0,n] = slope #stores the slope variable (the first derivative)
        txm[0,n] = np.mean(temp_regress_y) #change this to another statistic to calculate different rolling variables (standard deivation, percentiles, etc)
    txs[0,0:hws] = txs[0,hws] #handles the beginning points (sets them to the first processed point)
    txs[0,n_rows-hws:n_rows] = txs[0,n_rows-hws] #handles the end points (sets them to the last processed point)
    df_w['d_depth_dt'] = txs[0,:]
    rainmerge = pd.merge(rain,df_w, on = 'date_time')
    ax5.scatter(rainmerge.precip_in,rainmerge.d_depth_dt)
    ax5.set_title("x = rain, inches, y = d_depth/dt")    
    
    peak_thresh = 0.002
    
    df_w['peak'] = False
    df_w['peak'] = np.where(df_w['d_depth_dt'] > peak_thresh, True, False)
    
    ind_peaks = []    
    ind = 0;
    tog = df_w['peak'][0]            #Toggle is the previous value
    
    for index in range(len(df_w)):
        check = df_w['peak'][index]  # Check is the current value
        if check != tog:                # When a state change occurs
            if tog == True:             # See whether we're coming off a peak and if so:
                temp_peak = [ind,index]     # Add this peak to the list of peaks 
                ind_peaks = ind_peaks + [temp_peak]
                tog = check                 # Indicate that we're now at baseline (tog = check = False)
            if tog == False:            # If a state change occurs to come ONTO a peak
                ind = index                 # Save this position as the start of the peak
                tog = check                 # Set tog = check = True (we are now in a peak)
    
    df_peaks = pd.DataFrame(ind_peaks)
    df_peaks.columns = ['w_start','w_end'] # The dimensions of the rising limb or wetting limb of each peak
    df_peaks['w_dur'] = df_peaks.w_end-df_peaks.w_start # The length of each wetting event (rising limb)
    df_peaks['d_end'] = df_peaks['w_start'].shift(-1).fillna(df_peaks['w_end'].iloc[-1]).astype(int) # The end of the drying curve (falling limb)
    df_peaks['wd_dur'] = df_peaks.d_end-df_peaks.w_start # Return time of events (time from wetting event to next wetting event)
    # Now we have a dataframe of the peak start and end times from which we can extract some information
    subset = df_peaks[['w_start','d_end','w_end']]
    tuples = [x for x in subset.values.tolist()]
    # Now we have a list of lists containing the peak start, peak max, and start of the following peak
    
    #Analyzing and plotting peaks
    print(str(len(tuples)) + " peaks found")
    
    print("Analyze peaks")
    col2 = pl.cm.viridis(np.arange(365)/(365*1.0))
    
    for group in tuples[0:-2]:
        drow = [sensor_id]
        snip = df_w[['run_time','depth_ave','d_depth_dt']].loc[group[0]:group[1]].copy()
        y_0 = snip.depth_ave.reset_index(drop=True).iloc[0]
        y_peak = snip.depth_ave.loc[group[2]]
        y_max = snip.depth_ave.max()
        y_min = snip.depth_ave.min()
        y_rise = y_max - y_0
        y_rang = y_max - y_min
        
        x_0 = snip.run_time.reset_index(drop=True).iloc[0]
        x_max = snip.run_time.max()
        x_peak = snip.run_time.loc[group[2]]
        x_ymx = snip.run_time.loc[snip['depth_ave'] == y_max].item()
        x_rise = x_peak - x_0
        x_rang = x_max - x_0
        x_year = x_0/(3600*24)/365
        x_day = int((x_year - int(x_year))*365)
        
        drow = drow + [y_0,y_peak,y_max,y_min,y_rise,y_rang]
        drow = drow + [x_0,x_max,x_peak,x_ymx,x_rise,x_rang,x_day]    
        
        # x and y at the origin so that they all start at the same place
        snip['depth_norm1'] = (snip.depth_ave - y_0)
        snip['rt_norm1'] = (snip.run_time - x_0)
        
        # peak (x,y) at the origin
        snip['depth_norm2'] = (snip.depth_ave - y_max)
        snip['rt_norm2'] = (snip.run_time - x_ymx)
        
        # x0,y0 at origin, rising limb scaled to align
        snip['depth_norm3'] = (snip.depth_ave - y_0)/y_rise
        snip['rt_norm3'] = (snip.run_time - x_0)/x_rise
        
        snip['depth_norm4'] = (snip.depth_ave - y_0)/y_max
        snip['rt_norm4'] = (snip.run_time - x_0)/x_rang
        
        r_popt = r_pcov = f_popt = f_pcov = 0
# =============================================================================
#         # Regression: expn_func (y = a^x + b) , powr_func(y = k*x^n + b), linr_func (y = mx+b)
#         # Regression on rising limb
#         # TODO: Get somewhere to store these values
#         # TODO: Correlate antecedent moisture, precipitation
#         # TODO: Seasonality?
#         r_subsnip = snip[['rt_norm1','depth_norm1']].loc[group[0]:group[2]].copy()
#         r_popt,r_pcov = sc.optimize.curve_fit(expn_func,r_subsnip['rt_norm1'],r_subsnip['depth_norm1'])
#         drow = drow + [r_popt,r_pcov]
#     
#         # Regression on falling limb
#         f_subsnip = snip[['rt_norm1','depth_norm1']].loc[group[2]:group[1]].copy()
#         f_popt,f_pcov = sc.optimize.curve_fit(expn_func,f_subsnip['rt_norm1'],f_subsnip['depth_norm1'])
#         drow = drow + [f_popt,f_pcov]
# =============================================================================
        drow = drow + [np.nan,np.nan,np.nan,np.nan]
        if (y_max < 1) and (x_rang/(3600*24) < 30):
            ax1.plot(snip['run_time']/(3600*24),snip['depth_ave'],color = col2[x_day]) 
            ax2.plot(snip['run_time']/(3600*24),snip['d_depth_dt'],color = col2[x_day])
            ax3.plot(snip['rt_norm1']/(3600*24),snip['depth_ave'],color = col2[x_day])
            ax4.plot(snip['rt_norm4'],snip['depth_norm4'],color = col2[x_day])
        dtable = dtable + [drow]
    print("\n")
    
df_peakstats = pd.DataFrame(dtable)
df_peakstats.columns = ['sensor_id','y_0','y_peak','y_max','y_min','y_rise','y_rang','x_0','x_max','x_peak','x_ymx','x_rise','x_rang','x_day','r_popt','r_pcov','f_popt','f_pcov']  

# Don't run this again! I pickled it!
#df_peakstats.to_pickle("Pickled_df")
    
#fig1.suptitle(sensor_id + ' First derivative')
#fig2.suptitle(sensor_id + ' First derivative')


# Second derivative
txs=np.zeros(shape=[1,n_rows]) #temporary storage variable for the slope
txm=np.zeros(shape=[1,n_rows]) #temporary storage variable for the mean

for n in range(hws, n_rows - hws): ##
    temp_regress_y = df_w.d_depth_dt[n-hws:n+hws] #selects data to regress, could reduce noise even further by selecting the smoothed data from above.
    temp_regress_x = df_w.run_time[n-hws:n+hws] # x-coordinates for regression
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(temp_regress_x/(60.0**2),temp_regress_y) #linear regression
    txs[0,n] = slope #stores the slope variable (the first derivative)
    txm[0,n] = np.mean(temp_regress_y) #change this to another statistic to calculate different rolling variables (standard deivation, percentiles, etc)
txs[0,0:hws] = txs[0,hws] #handles the beginning points (sets them to the first processed point)
txs[0,n_rows-hws:n_rows] = txs[0,n_rows-hws] #handles the end points (sets them to the last processed point)
df_w['d2_depth_dt2'] = txs[0,:]

peak_thresh2 = 0.000

df_w['peak2'] = False
df_w['peak2'] = np.where(df_w['d2_depth_dt2'] > peak_thresh2, True, False)

ind_peaks = []    
ind = 0;
tog = df_w['peak2'][0]            #Toggle is the previous value

for index in range(len(df_w)):
    check = df_w['peak2'][index]  # Check is the current value
    if check != tog:                # When a state change occurs
        if tog == True:             # See whether we're coming off a peak and if so:
            temp_peak = [ind,index]     # Add this peak to the list of peaks 
            ind_peaks = ind_peaks + [temp_peak]
            tog = check                 # Indicate that we're now at baseline (tog = check = False)
        if tog == False:            # If a state change occurs to come ONTO a peak
            ind = index                 # Save this position as the start of the peak
            tog = check                 # Set tog = check = True (we are now in a peak)

df_peaks = pd.DataFrame(ind_peaks)
df_peaks.columns = ['w_start','w_end'] # The dimensions of the rising limb or wetting limb of each peak
df_peaks['w_dur'] = df_peaks.w_end-df_peaks.w_start # The length of each wetting event (rising limb)
df_peaks['d_end'] = df_peaks['w_start'].shift(-1).fillna(df_peaks['w_end'].iloc[-1]).astype(int) # The end of the drying curve (falling limb)
df_peaks['wd_dur'] = df_peaks.d_end-df_peaks.w_start # Return time of events (time from wetting event to next wetting event)
# Now we have a dataframe of the peak start and end times from which we can extract some information
subset = df_peaks[['w_start','d_end','w_end']]
tuples2 = [x for x in subset.values.tolist()]
# Now we have a list of lists containing the peak start, peak max, and start of the following peak


fig3, (ax6,ax7) = plt.subplots(2,1,sharex = True)
fig4, (ax9,ax10) = plt.subplots(2,1)
for group in tuples2[0:-2]:
    snip = df_w[['run_time','depth_ave','d_depth_dt','d2_depth_dt2']].loc[group[0]:group[1]].copy()
    y_max = snip.depth_ave.max()
    y_min = snip.depth_ave.min()
    y_0 = snip.depth_ave.reset_index(drop=True).iloc[0]
    y_rise = y_max - y_0
    y_range = y_max - y_min
    
    x_0 = snip.run_time.reset_index(drop=True).iloc[0]
    x_peak = snip.run_time.loc[group[2]]
    x_rise = x_peak - x_0
    x_max = snip.run_time.max()
    x_year = x_0/(3600*24)/365
    x_day = int((x_year - int(x_year))*365)
    x_range = x_max - x_0 
    
    snip['depth_norm1'] = (snip.depth_ave - y_0)
    snip['rt_norm1'] = (snip.run_time - x_0)
    snip['depth_norm2'] = (snip.depth_ave - y_0)/y_rise
    snip['rt_norm2'] = (snip.run_time - x_0)
    if (y_max < 1) and (x_range/(3600*24) < 20) :
        ax6.plot(snip['run_time']/(3600*24),snip['depth_ave'],color = col2[x_day]) 
        ax7.plot(snip['run_time']/(3600*24),snip['d2_depth_dt2'],color = col2[x_day])
        ax9.plot(snip['rt_norm1'],snip['depth_norm1'],color = col2[x_day])
        ax10.plot(snip['rt_norm2'],snip['depth_norm2'],color = col2[x_day])
fig3.suptitle('Second derivative')
fig4.suptitle('Second derivative')

plt.figure()
plt.subplot(1,2,1)
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_m,'b.'),plt.xlabel('Time (days)'), plt.ylabel('depth')
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_ave,'r-')
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_m.rolling(window_size,center=True,win_type='triang').mean(),'-',color='seagreen')
#plt.axis([112,114,.65,.85])#zooming in to see the difference between smoothed and raw data.
plt.subplot(1,2,2)
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_m,'b.'),plt.xlabel('Time (days)')
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_ave,'r-')
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_m.rolling(window_size,center=True,win_type='triang').mean(),'-',color='seagreen')
#plt.axis([112.5,113,.8,.86])

plt.figure()
plt.subplot(2,1,1)
plt.plot(df_w.run_time/(3600*24.0),df_w.depth_m,'b-'), plt.ylabel('depth (m)')
#plt.axis([112,114,.65,.9])
plt.subplot(2,1,2)
plt.plot(df_w.run_time/(3600*24.0),df_w.peak/20)
plt.plot(df_w.run_time/(3600*24.0),txs[0,:],'--',color='red'),plt.xlabel('Time (days)'),plt.ylabel('rate of change (m/hr)')
#plt.axis([112,114,-.1,.1])

