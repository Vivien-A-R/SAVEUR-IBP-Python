# -*- coding: utf-8 -*-
"""
Takes text-based data (boring logs) and assigns numeric (integer) codes to each unique value
For PCA or other numeric/categorical analysis
Created on Fri May 11 12:43:50 2018

Modfied peak analysis - takes a pre-processed data file

@author: Vivien
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as scs
from scipy.optimize import curve_fit as cf
from datetime import datetime
#from sklearn.decomposition import PCA
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)
pd.set_option('display.max_columns', 500)

# Import data
root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\'
data_path = root_path + '\\Processed Water Level Data\\'
proc_path = root_path + 'Water Level Derived Products\FilledSmoothed\\'
paperII_path = "C:\Users\Packman-Field\Documents\Paper II\Water Data\\"

#Define forms of functions for fitting purposes
def expn_func(x,a,b):
    return a**x + b

def powr_func(x,k,n,b):
    return k*x**n + b

def linr_func(x,m,b):
    return m*x+b

col2 = cm.get_cmap("hsv")

s = ["WLW1","WLW2","WLW3","WLW4","WLW5","WLW6","WLW7","WLW8","WLW9","WLW10","WLW12","WLW13"] #Figure out what's up with 14
#s = ["WLW10"]

pt = 0.002 #required threshold of slope in one timestep

fig2, (ax3,ax4) = plt.subplots(2,1, sharex = True)

# Takes a Pandas dataframe, returns list of tuples representing the indices of peaks in that data frame
def peakfind():
    # Dummy data (WLW5) remove this line later and move to function arguments
    df_w=pd.read_csv(paperII_path + "\\waterlevels\\" + "WLW5" + '_' + fs_suffix + '.csv' , parse_dates=['date_time'])
    #Get rid of extra information
    df_w.drop(['pressure_pa','temperature_c','WS_elevation_m'],axis = 1, inplace = True)
    df_w.rename({'depth_m_smoothed':'depth_ave'}, axis = 1, inplace = True)
    n_rows, n_cols = df_w.shape
    
    print("Finding peaks")
    # Find peaks
    #def sl(oneroll):
    #    #do regression on the rolling window
    #    return slope
    #rolls = df_w[['run_time','depth_ave']].rolling(window = 8,center = True)
    #df_w['slope'] = rolls.apply(sl)
    #pandas rolling window apply does NOT work here because in this package version it only takes one column. Boo.
    # TODO: Update Pandas version and rewrite this neatly
    
    # Set up rolling window params to obtain slopes
    roll_ws = 8
    hws = roll_ws/2 # half the size of the window over which the regression will work.
    
    # First derivative
    txs=np.zeros(shape=[1,n_rows]) #temporary storage variable for the slope
    txm=np.zeros(shape=[1,n_rows]) #temporary storage variable for the mean
    
    for n in range(hws, n_rows - hws): ##
        temp_regress_y = df_w.depth_ave[n-hws:n+hws] #selects data to regress
        temp_regress_x = df_w.run_time[n-hws:n+hws] # x-coordinates for regression
        slope, intercept, r_value, p_value, std_err = scs.linregress(temp_regress_x/(60.0**2),temp_regress_y) #linear regression
        txs[0,n] = slope #stores the slope variable (the first derivative)
        txm[0,n] = np.mean(temp_regress_y) #change this to another statistic to calculate different rolling variables (standard deivation, percentiles, etc)
    txs[0,0:hws] = txs[0,hws] #handles the beginning points (sets them to the first processed point)
    txs[0,n_rows-hws:n_rows] = txs[0,n_rows-hws] #handles the end points (sets them to the last processed point)
    df_w['d_depth_dt'] = txs[0,:]  
    
    df_w['peak'] = False
    df_w['peak'] = np.where(df_w['d_depth_dt'] > peak_thresh, True, False) # Find the starts of peaks based on slope, set True
    
    ind_peaks = []    
    ind_a = 0;
    tog = df_w['peak'][0]                   #Toggle is the previous value (bool), initially False
    
    for ind_b in range(len(df_w)):
        check = df_w['peak'][ind_b]         # Check is the current value to be checked against tog
        if check != tog:                    # When a state change occurs
            if tog == True:                 # Previous values are from inside a peak
                temp_peak = [ind_a,ind_b]   # Add this peak to the list of peaks
                ind_peaks = ind_peaks + [temp_peak]
                tog = check                 # Indicate that we're now at baseline (tog = check = False)
            if tog == False:                # If a state change occurs to come ONTO a peak
                ind_a = ind_b               # Save this position as the start of the peak
                tog = check                 # Set tog = check = True (we are now in a peak)
    
    df_peaks = pd.DataFrame(ind_peaks)
    df_peaks.columns = ['w_start','w_end'] # The dimensions of the rising limb or wetting limb of each peak
    df_peaks['d_end'] = df_peaks['w_start'].shift(-1).fillna(df_peaks['w_end'].iloc[-1]).astype(int) # The end of the drying curve (falling limb) including dry tails
   # Now we have a dataframe of the peak start and end times from which we can extract some information

    subset = df_peaks[['w_start','w_end','d_end']]
    tuples = [x for x in subset.values.tolist()]    # Now we have a list of lists containing the wetting start, wetting end, drying end (in that order)
    print(str(len(tuples)) + " peaks found")        
    
    return tuples

def peakparams():
    dtable = []
    
def peakfit():
    dtable = []

# Splitting this into separate functions
def peakstats(sensors = s, fs_suffix = 'f5_lowess', peak_thresh = 0.01):
    # Which filled smoothed variant do we want (should be created by running the smooth_lowess_rect.py script (integrate these function calls into current script or create a module to call from other script)) 
    #fig1, (ax1,ax2) = plt.subplots(2,1,sharex = True)
    
    dtable = []
    ftable = []
    for sensor_id in sensors:
        print(sensor_id)
        df_w=pd.read_csv(paperII_path + "\\waterlevels\\" + sensor_id + '_' + fs_suffix + '.csv' , parse_dates=['date_time'])
    
        #Get rid of extra information
        df_w.drop(['pressure_pa','temperature_c','WS_elevation_m'],axis = 1, inplace = True)
        df_w.rename({'depth_m_smoothed':'depth_ave'}, axis = 1, inplace = True)
        n_rows, n_cols = df_w.shape
        
        print("Finding peaks")
        # Find peaks
        
        #def sl(oneroll):
        #    #do regression on the rolling window
        #    return slope
        #rolls = df_w[['run_time','depth_ave']].rolling(window = 8,center = True)
        #df_w['slope'] = rolls.apply(sl)
        #pandas rolling window apply does NOT work here because this version it only takes one column. Boo.
        # TODO: Update Pandas version and rewrite this neatly
        
        # Set up rolling window params to obtain slopes
        roll_ws = 8
        hws = roll_ws/2 # half the size of the window over which the regression will work.
        
        # First derivative
        txs=np.zeros(shape=[1,n_rows]) #temporary storage variable for the slope
        txm=np.zeros(shape=[1,n_rows]) #temporary storage variable for the mean
        
        for n in range(hws, n_rows - hws): ##
            temp_regress_y = df_w.depth_ave[n-hws:n+hws] #selects data to regress
            temp_regress_x = df_w.run_time[n-hws:n+hws] # x-coordinates for regression
            slope, intercept, r_value, p_value, std_err = scs.linregress(temp_regress_x/(60.0**2),temp_regress_y) #linear regression
            txs[0,n] = slope #stores the slope variable (the first derivative)
            txm[0,n] = np.mean(temp_regress_y) #change this to another statistic to calculate different rolling variables (standard deivation, percentiles, etc)
        txs[0,0:hws] = txs[0,hws] #handles the beginning points (sets them to the first processed point)
        txs[0,n_rows-hws:n_rows] = txs[0,n_rows-hws] #handles the end points (sets them to the last processed point)
        df_w['d_depth_dt'] = txs[0,:]  
        
        df_w['peak'] = False
        df_w['peak'] = np.where(df_w['d_depth_dt'] > peak_thresh, True, False) # Find the starts of peaks based on slope, set True
        
        ind_peaks = []    
        ind_a = 0;
        tog = df_w['peak'][0]                   #Toggle is the previous value (bool), initially False
        
        for ind_b in range(len(df_w)):
            check = df_w['peak'][ind_b]         # Check is the current value to be checked against tog
            if check != tog:                    # When a state change occurs
                if tog == True:                 # Previous values are from inside a peak
                    temp_peak = [ind_a,ind_b]   # Add this peak to the list of peaks
                    ind_peaks = ind_peaks + [temp_peak]
                    tog = check                 # Indicate that we're now at baseline (tog = check = False)
                if tog == False:                # If a state change occurs to come ONTO a peak
                    ind_a = ind_b               # Save this position as the start of the peak
                    tog = check                 # Set tog = check = True (we are now in a peak)
        
        df_peaks = pd.DataFrame(ind_peaks)
        df_peaks.columns = ['w_start','w_end'] # The dimensions of the rising limb or wetting limb of each peak
        df_peaks['d_end'] = df_peaks['w_start'].shift(-1).fillna(df_peaks['w_end'].iloc[-1]).astype(int) # The end of the drying curve (falling limb) including dry tails
       # Now we have a dataframe of the peak start and end times from which we can extract some information

        subset = df_peaks[['w_start','w_end','d_end']]
        tuples = [x for x in subset.values.tolist()]    # Now we have a list of lists containing the wetting start, wetting end, drying end (in that order)
        print(str(len(tuples)) + " peaks found")        
        
        #Analyzing and plotting peaks
        print("Analyze peaks")
        
        for peak_inds in tuples[0:-2]:
            
            snip = df_w[['date_time','run_time','depth_ave','sensor_elev_m','d_depth_dt']].loc[peak_inds[0]:peak_inds[2]].copy()
            snip = snip[snip.depth_ave > 0.05]
            y_max = snip.depth_ave.max() # Max water depth over interval
            
            if (len(snip) > 48 and y_max < 1): #more than 24 hours and gw only
                y_peak = snip.depth_ave.loc[peak_inds[1]] # Water depth of identified peak
                y_error = y_peak - y_max # Vertical difference between algorithmically-identified "peak" and max value in range
                
                x_peak = snip.run_time.loc[peak_inds[1]] # 
                x_ymx = snip.run_time.loc[snip['depth_ave'] == y_max].iloc[0].item() # Time of max y
                x_error = (x_peak-x_ymx)/(3600*24) # Time difference between algorithmically-identified "peak" and time of max y
    
                if (x_error < 20 and x_error > -20 and y_error > -0.2): # Skip the lumpy peaks           
                    
                    snip = snip.reset_index(drop=True)
                    
                    y_0 = snip.depth_ave.iloc[0]
                    y_min = snip.depth_ave.min()
                    y_sensor = snip.sensor_elev_m.iloc[0]
                    
                    x_0 = snip.run_time.iloc[0] # First time step in peak
                    x_0_datetime = snip.date_time.iloc[0] # Datetime of first time step in peak
                    x_max = snip.run_time.max() # Last time step in peak
                    x_day = x_0_datetime.dayofyear/365.0
                    
                    drow = ([sensor_id] + 
                            [x_0,x_0_datetime,x_max,x_peak,x_ymx,x_day] + 
                            [y_sensor] + [y_0,y_peak,y_max,y_min])
                    dtable = dtable + [drow]
                    
                    snip['rt_norm1'] = (snip.run_time - x_0)    # x at the origin
                    snip['depth_norm2'] = (snip.depth_ave - y_peak) # peak (x,y) at the origin
                    snip['rt_norm2'] = (snip.run_time - x_peak) # peak (x,y) at the origin
                    
                    ax3.plot(snip['rt_norm1']/(3600*24),snip['depth_ave'],color = col2(x_day),alpha = 0.5)
                    ax4.plot(snip['rt_norm2']/(3600*24),snip['depth_norm2'],color = col2(x_day), alpha = 0.5)
                
                    # Regression: expn_func (y = a^x + b) , powr_func(y = k*x^n + b), linr_func (y = mx+b)
                    
                    # Regression on falling limb
                    f_subsnip = snip[['rt_norm1','depth_norm2']].loc[peak_inds[2]:peak_inds[1]].copy()
                    f_popt,f_pcov = cf(expn_func,f_subsnip['rt_norm1'],f_subsnip['depth_norm2'])
                    
                    frow = [f_popt,f_pcov]
                    ftable = ftable + [frow]

        print("Screened and tidied, " + str(len(dtable)) + " total peaks analyzed")
        print("\n")
        
    df_peakstats = pd.DataFrame(dtable) #Combine the list (dtable) of lists (drows) to make a data frame
    df_peakstats.columns = ['sensor_id','x_0','x_0_dt','x_max','x_peak','x_ymx','x_day','y_sensor','y_0','y_peak','y_max','y_min']  
    
    #ftable = [[np.nan, np.nan]] #Dummy while I figure out the fits.
    df_fitstats = pd.DataFrame(ftable)
    df_fitstats.columns = ['falling_params','falling_cov']
    
    return df_peakstats, df_fitstats

df_p,df_f = peakstats(peak_thresh = pt)

#df_p.to_csv(paperII_path + "peakstats_thresh " + str(pt) + "m.csv")
#df_f.to_csv(paperII_path + "fitstats_thresh " + str(pt) + "m.csv")

# =============================================================================
# rain = pd.read_csv(rain_path + "PRECIP_IBP_filled_hrly.csv",index_col = 0,parse_dates = ['date_time'])
# ps_nofit = df_p[['sensor_id','x_0','y_0','x_peak','y_peak','x_max','x_ymx','y_max','y_min','y_rise','y_rang','x_rise','x_rang','x_day']].copy()
# #ps_nofit.to_csv("C:\Users\Packman-Field\Google Drive\Packman Group\Python Scripts\IBP_scripts\\data_products\\202105_peakstats.csv")
# ps_nofit['x_0'] = ps_nofit['x_0'].astype(int)
# ps_nofit['sensor_id'] = pd.to_numeric(ps_nofit.sensor_id.str.replace(r'\D+', ''))
# 
# str_start = "2016-07-09 11:00:00"
# dt_start = datetime.strptime(str_start,"%Y-%m-%d %H:%M:%S")
# 
# #redo this with Crete data
# rain['P12_m'] = rain.precip_in.rolling(12).sum()*0.0254
# rain['P24_m'] = rain.precip_in.rolling(24).sum()*0.0254
# rain.dropna(inplace = True)
# rain['P24_m'] = rain['P24_m'].round(3)
# rain['P12_m'] = rain['P12_m'].round(3)
# 
# rain['rt'] = rain['date_time']-dt_start
# rain['rt_sec'] = rain.rt.dt.total_seconds().astype(int)
# rain = rain[rain['rt_sec']>=0]
# rain = rain.set_index('rt_sec')
# 
# ps_nofit = ps_nofit.merge(rain[['P12_m','P24_m']],left_on='x_0',right_index=True,how = 'left')
# 
# ps_trim = ps_nofit[(ps_nofit != 0).all(1)].dropna()
# ps_nofit['x_day'] = ps_nofit['x_day']/365
# 
# ps_nofit.plot.scatter(x = 'y_rise',y='P24_m',s=50,c = 'x_day',colormap = 'hsv',logx=False)
# ps_nofit.plot.scatter(x = 'y_rise',y='P24_m',s=50,c = 'x_day',colormap = 'hsv',logx=True,logy=True)
# ps_nofit.plot.scatter(x = 'y_rise',y='P24_m',s=50,c = 'sensor_id',colormap = 'viridis',logx=True,logy=True)
# ps_nofit.plot.scatter(x = 'y_rise',y='P24_m',s=50,c = 'sensor_id',colormap = 'viridis',logx=False,logy=False)
# 
# fpca = plt.figure()
# ax = plt.gca()
# pca = PCA(2)
# projected = pca.fit_transform(ps_nofit)
# ax.scatter(projected[:, 0], projected[:, 1],
#             s= 50, c=ps_nofit.x_day, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('spectral', 10))
# ax.set_xlabel('component 1')
# ax.set_ylabel('component 2')
# #plt.colorbar(ax=ax)
# =============================================================================

#sns.clustermap(df_p)
#    sns.pairplot(ps_nofit,hue = "sensor_id")

