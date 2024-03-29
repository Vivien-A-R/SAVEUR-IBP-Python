# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:26:57 2017

@author: Vivien


"""
import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.signal
from astropy.stats import LombScargle
import statsmodels.nonparametric.smoothers_lowess as sml
from scipy.interpolate import interp1d

# TODO: Update this for PII
root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\'
data_path = root_path + 'Processed Water Level Data\\'
rain_data = root_path + 'Precipitation\\'
soil_path = root_path + 'Data From SMP\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
sensor_meta = pd.read_table(data_path+'wl_position_meta.csv',sep=',',index_col=False)


fmax = 10 * 365.25 * 24                 # 10 years
fmin = 12.0              # 12 hours
frequency = np.linspace(1.0/fmax, 1.0/fmin, 1000000) #half-day to 10 year intervals, 1 million steps 
wavelength_day = frequency**-1/24.0
       
def wl_spectral(sensor_id):
    datafile = pd.read_csv(data_path + sensor_id + "_ibp_main.csv")
    x_f  = datafile.run_time[0:len(datafile.run_time)/2]/(datafile.run_time.iloc[-1]*60*30) #Generate frequencies from time range of data
    y_p  = scipy.fftpack.fft(datafile.WS_elevation_m)
    
    #Get the appropriate things to plot and calculate regressions
    N = len(y_p)
    y_pn_arr = 2.0/N*np.abs(y_p[0:N/2])**2
    y_pn = pd.Series(y_pn_arr)
    y_pn_log = np.log10(y_pn_arr[1:])
    x_f_arr= np.array(x_f)
    x_f_log = np.log10(x_f_arr[1:])
    
    #Get loglog regression
    A = np.vstack([x_f_log, np.ones(len(x_f_log))]).T
    m,c = np.linalg.lstsq(A,y_pn_log)[0] #Slope, intercept of loglog regression
    print("slope: " + '{:04.3f}'.format(m) + " int: " + '{:04.2f}'.format(c))
    
    # returns x, y power spectra, slope-intercept for loglog regression
    return x_f,y_pn,m,c

#LS on rainfall
# Set up time-series
rain_NOAA = pd.read_csv(rain_data + "PRECIP_NOAA_hrly.csv",index_col = 0,parse_dates = ['date_time'])
rain_NOAA['cumsum'] = rain_NOAA.precip_in_Midway.cumsum()
rain_column = 'precip_in_Midway'  # Change this to easily swap between cumulative and non-cumulative time series

step = len(rain_NOAA)-1
rain_startind = 0
while rain_startind + step < len(rain_NOAA):
    rain_endind = rain_startind + step
    df_precip = rain_NOAA[rain_startind:rain_endind].reset_index(drop = True).copy() # Make a copy with just the data we need
    df_precip['t_delta'] = (df_precip.date_time - df_precip.date_time.iloc[0]).dt.total_seconds() # Create a cumulative time column
    #df_precip = df_precip[(df_precip != 0).all(1)] # Drop zeroes (optional, does better without them!)
    df_precip.dropna(inplace = True)        # Remove rows with nan
    rain_startind = rain_startind + step
    
    # Set up input arrays and model parameters
    x = np.array(df_precip.t_delta)
    t = x/3600    #units in hours to get rid of "hourly" signal
    y = np.array(df_precip[rain_column])    # Choose which column
    y_detrend = scipy.signal.detrend(y)
    fmax = 10 * 365.25 * 24                 # 10 years
    fmin = 12.0              # 12 hours
    frequency = np.linspace(1.0/fmax, 1.0/fmin, 1000000) #half-day to 10 year intervals, 1 million steps
    period = frequency**-1/24.
    power = LombScargle(t,y_detrend).power(frequency)
    
    # Plot
    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(x, y_detrend,marker ='+',rasterized = True)
    plt.subplot(2,1,2)
    plt.loglog(period,power)
    plt.suptitle("Precipitation (78 years of Midway)")
    
    annfreq = (frequency[np.argmax(power)]**-1)/24/365.25
    print("Max power in rain signal at: " + 
          "{0:0.4f}".format(annfreq) + " years")

# =============================================================================
# # New smoothing routine: LOWESS
# lowess_xy = sml.lowess(y,x,0.0005,delta = 0.0001*max(x))
# lowess_x = list(zip(*lowess_xy))[0]
# lowess_y = list(zip(*lowess_xy))[1]
# pd.DataFrame(lowess_xy).plot(x=0)
# pd.DataFrame([x,y]).transpose().plot(x=0)
# f = interp1d(lowess_x, lowess_y, bounds_error=False)
# xnew = range(int(x[0]),int(x[-1]),1800)
# ynew = f(xnew)
# plt.figure()
# plt.plot(x, y, 'o')
# plt.plot(lowess_x, lowess_y, '*')
# plt.plot(xnew, ynew, '-')
# plt.show()
# =============================================================================

# LS on WL
# Use this one to find the rollover point
def ls_well(well_id):
    # Set up time-series
    wl_signal = pd.read_csv(data_path + well_id + "_ibp_main.csv",
                            index_col = None,
                            parse_dates = ['date_time'])
    wl_signal['t_delta'] = (wl_signal.date_time - wl_signal.date_time.iloc[0]).dt.total_seconds() # Create a cumulative time column
    wl_signal = wl_signal[wl_signal.qual_c == 1] # Drop flagged data
    wl_signal.dropna(inplace = True)        # Remove rows with nan
    gelev_m = sensor_meta[sensor_meta['sensor'] == well_id].ground_elev_ft.drop_duplicates().item()*0.3048
    
    
    # Set up input arrays and model parameters
    x = np.array(wl_signal.t_delta)
    t = x/(3600.)
    y = np.array(wl_signal.WS_elevation_m - gelev_m)             # Choose which column
    #y_detrend = scipy.signal.detrend(y)
    power = LombScargle(t,y).power(frequency)   # Frequency set above in rain series analysis
    
    annfreq = (frequency[np.argmax(power)]**-1)/24/365.25
    print("Max power in " + well_id + " signal at: " + 
          "{0:0.4f}".format(annfreq) + " years")
    
    N = len(power)
    pn_l = np.log10(power)
    freq_l = np.log10(frequency)
    A = np.vstack([freq_l, np.ones(len(freq_l))]).T
    cos = np.logspace(3.5,np.log10(0.5*N),100,dtype = int,endpoint = False)
    rsq_left = []
    rsq_right = []
    for co in cos:
        pn_log = pn_l[1:co]
        regress = np.linalg.lstsq(A[1:co],pn_log) #Slope, intercept of loglog regression
        m1, b1 = regress[0]
        rsq1 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
        rsq_left = rsq_left + [rsq1]
        
        pn_log = pn_l[co:]
        regress = np.linalg.lstsq(A[co:],pn_log) #Slope, intercept of loglog regression
        m2, b2 = regress[0]
        rsq2 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
        rsq_right = rsq_right + [rsq2]
    rsq_sum = [a+b for a,b in zip(rsq_left,rsq_right)]
    
    return cos,rsq_sum
# =============================================================================
# sids1 = ["WLW7","WLW8","WLW9","WLW10"]
# 
# # Find the rollover point
# test1 = []
# test2 = []
# test3 = []
# for sid in sids1:
#     c,s = ls_well(sid)
#     
#     test = pd.DataFrame([frequency[c],s]).transpose()
#     rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
#     print("Rollover in " + sid + " power spectra at " +
#           "{0:0.2f}".format(rollover) + " days.")
#     
#     test1 = test1 + [frequency[c]]
#     test2 = test2 + [s]
#     test3 = test3 + [c]
# 
# test = pd.DataFrame([pd.DataFrame(test1).mean(),
#                      pd.DataFrame(test2).mean()]).transpose()
# rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
# print("Average rollover in power spectra at " +
#       "{0:0.2f}".format(rollover) + " days.")
# 
# breakpoint = 1/(rollover * 24)
# =============================================================================

breakpoint = 0.00351434828961637

# Use this one to plot
def ls_well2(well_id):
    # Set up time-series
    wl_signal = pd.read_csv(data_path + well_id + "_ibp_main.csv",
                            index_col = None,
                            parse_dates = ['date_time'])
    wl_signal['t_delta'] = (wl_signal.date_time - wl_signal.date_time.iloc[0]).dt.total_seconds() # Create a cumulative time column
    wl_signal = wl_signal[wl_signal.qual_c == 1] # Drop flagged data
    wl_signal.dropna(inplace = True)        # Remove rows with nan
    gelev_m = sensor_meta[sensor_meta['sensor'] == well_id].ground_elev_ft.drop_duplicates().item()*0.3048
    
    # Set up input arrays and model parameters
    x = np.array(wl_signal.t_delta)
    t = x/(3600.)
    y = np.array(wl_signal.WS_elevation_m - gelev_m)             # Choose which column
    #y_detrend = scipy.signal.detrend(y)
    power = LombScargle(t,y).power(frequency)   # Frequency set above in rain series analysis
    
    dayfreq = (frequency[np.argmax(power)]**-1)/24.0
    annfreq = dayfreq / 365.25
    report = "Max power in " + well_id + " signal at: " +  "{0:0.4f}".format(annfreq) + " years or " +  "{0:0.1f}".format(dayfreq) + " days."
    
    pn_l = np.log10(power)
    freq_l = np.log10(frequency)
    A = np.vstack([freq_l, np.ones(len(freq_l))]).T
    l_ind = np.where(frequency < bkp_w)
    r_ind = np.where(frequency > bkp_w)
    
    pn_log = pn_l[l_ind]
    regress = np.linalg.lstsq(A[l_ind],pn_log) #Slope, intercept of loglog regression
    ml, bl = regress[0]
    #rsq1 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
    
    pn_log = pn_l[r_ind]
    regress = np.linalg.lstsq(A[r_ind],pn_log) #Slope, intercept of loglog regression
    mr, br = regress[0]
    print(well_id + " slope: " + "{0:0.4f}".format(mr))
    #rsq2 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
    
    #Plot
    plt.figure()
    ax1 = plt.subplot(2,1,1)
    ax1.scatter(x, y)
    ax2 = plt.subplot(2,1,2)
    ax2.loglog(wavelength_day,power)
    #ax2.loglog(frequency[l_ind],10**(ml*freq_l[l_ind]+bl), 'r-')
    #ax2.loglog(frequency[r_ind],10**(mr*freq_l[r_ind]+br), 'r-')
    plt.suptitle(well_id)
    print report


#Redo this soon
# =============================================================================
# # Find the rollover point
# test1 = []
# test2 = []
# test3 = []
# for sid in sids2:
#     c,s = ls_well(sid)
#     
#     test = pd.DataFrame([frequency[c],s]).transpose()
#     rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
#     print("Rollover in " + sid + " power spectra at " +
#           "{0:0.2f}".format(rollover) + " days.")
#     
#     test1 = test1 + [frequency[c]]
#     test2 = test2 + [s]
#     test3 = test3 + [c]
# 
# test = pd.DataFrame([pd.DataFrame(test1).mean(),
#                      pd.DataFrame(test2).mean()]).transpose()
# rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
# print("Average rollover in power spectra at " +
#       "{0:0.2f}".format(rollover) + " days.")
# 
# breakpoint = 1/(rollover * 24)
# =============================================================================
bkp_w = 0.004068356327083226

   
def ls_smp(probe_id,sensor):
    sm_signal = pd.read_csv(soil_path + probe_id + '_ibp_main.csv',index_col = None, parse_dates = ['date_time'])
    sm_signal['t_delta'] = (sm_signal.date_time - sm_signal.date_time.iloc[0]).dt.total_seconds() # Create a cumulative time column
    sm_signal = sm_signal[sm_signal.qual_c == 1]    # Drop flagged data
    sm_signal.dropna(inplace = True)                # Remove rows with nan
    
    x = np.array(sm_signal.t_delta)
    t = x/3600.

    y = np.array(sm_signal[sensor + '_moisture'])
    power = LombScargle(t,y).power(frequency)
    
    annfreq = (frequency[np.argmax(power)]**-1)/24/365.25
    report = "Max power in " + probe_id + sensor + " signal at: " + "{0:0.5f}".format(annfreq) + " years"
    print report
    
    N = len(power)
    pn_l = np.log10(power)
    freq_l = np.log10(frequency)
    A = np.vstack([freq_l, np.ones(len(freq_l))]).T
    cos = np.logspace(3.5,np.log10(0.5*N),100,dtype = int,endpoint = False)
    rsq_left = []
    rsq_right = []
    for co in cos:
        pn_log = pn_l[1:co]
        regress = np.linalg.lstsq(A[1:co],pn_log) #Slope, intercept of loglog regression
        m1, b1 = regress[0]
        rsq1 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
        rsq_left = rsq_left + [rsq1]
        
        pn_log = pn_l[co:]
        regress = np.linalg.lstsq(A[co:],pn_log) #Slope, intercept of loglog regression
        m2, b2 = regress[0]
        rsq2 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
        rsq_right = rsq_right + [rsq2]
    rsq_sum = [a+b for a,b in zip(rsq_left,rsq_right)]
    
    return cos,rsq_sum
    
def ls_smp2(probe_id,sensor):
    sm_signal = pd.read_csv(soil_path + probe_id + '_ibp_main.csv',index_col = None, parse_dates = ['date_time'])
    sm_signal['t_delta'] = (sm_signal.date_time - sm_signal.date_time.iloc[0]).dt.total_seconds() # Create a cumulative time column
    sm_signal = sm_signal[sm_signal.qual_c == 1]    # Drop flagged data
    sm_signal.dropna(inplace = True)                # Remove rows with nan
    
    x = np.array(sm_signal.t_delta)
    t = x/3600.

    y = np.array(sm_signal[sensor + '_moisture'])
    power = LombScargle(t,y).power(frequency)
    
    pn_l = np.log10(power)
    freq_l = np.log10(frequency)
    A = np.vstack([freq_l, np.ones(len(freq_l))]).T
    l_ind = np.where(frequency < bkp_s)
    r_ind = np.where(frequency > bkp_s)
    
    pn_log = pn_l[l_ind]
    regress = np.linalg.lstsq(A[l_ind],pn_log) #Slope, intercept of loglog regression
    ml, bl = regress[0]
    #rsq1 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
    
    pn_log = pn_l[r_ind]
    regress = np.linalg.lstsq(A[r_ind],pn_log) #Slope, intercept of loglog regression
    mr, br = regress[0]
    #rsq2 = (1 - regress[1] / sum((pn_log - pn_log.mean())**2))[0]
    
    #Plot
    ax1 = plt.subplot(2,1,1)
    ax1.scatter(x, y,marker ='+')
    ax2 = plt.subplot(2,1,2)
    ax2.loglog(wavelength_day,power,rasterized=True,alpha = 0.3)
    #ax2.loglog(frequency[l_ind],10**(ml*freq_l[l_ind]+bl), 'r-',alpha = 0.3)
    #ax2.loglog(frequency[r_ind],10**(mr*freq_l[r_ind]+br), 'r-',alpha = 0.3)
    plt.suptitle(probe_id+sensor)
    
    annfreq = (frequency[np.argmax(power)]**-1)/24/365.25
    report = "Max power in " + probe_id + sensor + " signal at: " + "{0:0.5f}".format(annfreq) + " years"
    print report
        
# =============================================================================
# for p_id in probes:
#     test1 = []
#     test2 = []
#     test3 = []
#     for sens in sensors:
#         # Find the rollover point
#         c,s = ls_smp(p_id,sens)
#         
#         test = pd.DataFrame([frequency[c],s]).transpose()
#         rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
#         print("Rollover in " + p_id + sens + " power spectra at " +
#               "{0:0.2f}".format(rollover) + " days.")
#         
#         test1 = test1 + [frequency[c]]
#         test2 = test2 + [s]
#         test3 = test3 + [c]
#         
#     test = pd.DataFrame([pd.DataFrame(test1).mean(),
#                          pd.DataFrame(test2).mean()]).transpose()
#     rollover = (test[test[1] == test.max()[1]][0].item()**-1)/24
#     print("Average rollover in power spectra at " +
#           "{0:0.2f}".format(rollover) + " days.")
#     
#     breakpoint = 1/(rollover * 24)
# =============================================================================
bkp_s = 0.003496600701711357

# TODO: Get everything into comparable UNITS so we can compare magnitudes (Kendall and Hyndman, 2007)
# TODO: Find secondary peaks by finding the point on the right slope of the PSD that is furthest from the fit?
    

sids1 = ["WLW1","WLW2","WLW3","WLW4","WLW5","WLW6"] #wells
sids2 = ["WLS1","WLS2","WLS4","WLS6","WLS7","WLS8"] #surface

probes = ['SMP1','SMP2']
sensors = ['a1','a2','a3','a4','a5','a6']

for sid in sids1:
    ls_well2(sid)
    
# =============================================================================
# for sid in sids2:
#     ls_well2(sid)
# 
# =============================================================================
for p_id in probes:
    plt.figure()
    for sens in sensors:
        ls_smp2(p_id,sens)
        
