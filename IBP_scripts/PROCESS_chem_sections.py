# -*- coding: utf-8 -*-
"""
Documentation VAR 20181108:
    Assigns properties to each "Chem" section collected based on water time series
    - Use section recovery to "stretch" each section so that the full core is 120 cm
    - Use water time series to determine which each section was underwater
    - Also explore "toggling" frequency (wet-to-dry or dry-to-wet)

Created on Wed May 24 21:03:46 2017

@author: Vivien
"""
from __future__ import division  # Division that doesn't round down to the nearest integer
from meta_get import meta_get
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\\'
#Stretching samples to expand to full 30 cm
data_path = root_path + 'IBP Project\Documents\Processed Water Level Data\\'

#Get metadata for each sensor, including location of processed water level data file for each
sensor_meta = meta_get(data_path)

#Calculates the fraction of data points in a time-series where the water elevation is greater than a certain value
def fraction_wet(sensor_id,elevation):
    tpath = data_path+sensor_id+'_ibp_main.csv'
    datafile = pd.read_csv(tpath,index_col = 0,parse_dates=['date_time'])

    numgreater = (datafile['WS_elevation_m'] >= elevation).sum()
    numtotal = datafile.WS_elevation_m.size

    return float(numgreater)/float(numtotal)

#Does the same but averaging duplicate timestamps
def fraction_wet2(sensor_id,elevation):
    tpath = data_path+sensor_id+'_ibp_main.csv'
    datafile = pd.read_csv(tpath,parse_dates=['date_time'])

    wl_series = datafile[['date_time','WS_elevation_m']]
    oneyear = wl_series.groupby([lambda x : wl_series['date_time'][x].month,
                              lambda x: wl_series['date_time'][x].day,
                              lambda x: wl_series['date_time'][x].hour,
                              lambda x: wl_series['date_time'][x].minute]).mean()
    oneyear.reset_index(inplace = True)
    oneyear.columns = ['month','day','hour','minute','WS_elevation_m']
    oneyear['year']=1900
    oneyear['date_time']=pd.to_datetime(oneyear[['year', 'month', 'day', 'hour','minute']])
    oneyear.drop(['month','day','hour','year','minute'], inplace = True,axis = 1)
    
    
    numgreater = (oneyear['WS_elevation_m'] >= elevation).sum()
    numtotal = oneyear.WS_elevation_m.size
    
    return float(numgreater)/float(numtotal)

def toggle_count(sensor_id,elevation,par = 'p'):
    tpath = data_path+sensor_id+'_ibp_main.csv'
    datafile = pd.read_csv(tpath,index_col = None,parse_dates=['date_time'])
    datafile['WS_check'] = datafile['WS_elevation_m']*datafile['qual_c'].clip_lower(0)
    

    datafile['iswet'] = datafile['WS_check'] >= elevation
    #numtotal = datafile.WS_elevation_m.size
    datafile['block'] = (datafile.iswet.shift(1) != datafile.iswet).astype(int).cumsum()
    wets = datafile.reset_index().groupby(['iswet','block'])['index'].apply(np.array)
    
    e = datafile.date_time.max()
    s = datafile.date_time.min()
    ts_duration = (e - s).days* 60*60*24 + (e - s).seconds
    
    evertog = (len(wets.index.get_level_values(0).unique()) > 1 )
    
    if evertog == False: return np.nan
    
    else:
        lengths = [len(chunk) for chunk in wets[True]] 
        spans = len(lengths) #The number of continuous time spans of wetness or dryness
        durations = [dur*30*60 for dur in lengths] #The duration of those time periods (in seconds)
    
        toggle_per = ts_duration/(spans*2) #Average period of toggles between wet and dry 
    
        if(par == 'p'): return toggle_per
        elif(par == 'd'): return np.mean(durations)
    
# No-average calculation (could oversample some time points)
def fraction_dry(sensor_id,elevation):
    tpath = data_path+sensor_id+'_ibp_main.csv'
    datafile = pd.read_csv(tpath,index_col = 0,parse_dates=['date_time'])

    numgreater = (datafile['WS_elevation_m'] <= elevation).sum()
    numtotal = datafile.WS_elevation_m.size
    return float(numgreater)/float(numtotal)

# "Yearly-averaged" calculation
def fraction_dry2(sensor_id,elevation):
    tpath = data_path+sensor_id+'_ibp_main.csv'
    datafile = pd.read_csv(tpath,parse_dates=['date_time'])
    
    wl_series = datafile[['date_time','WS_elevation_m']]
    oneyear = wl_series.groupby([lambda x : wl_series['date_time'][x].month,
                              lambda x: wl_series['date_time'][x].day,
                              lambda x: wl_series['date_time'][x].hour,
                              lambda x: wl_series['date_time'][x].minute]).mean()
    oneyear.reset_index(inplace = True)
    oneyear.columns = ['month','day','hour','minute','WS_elevation_m']
    oneyear['year']=1900
    oneyear['date_time']=pd.to_datetime(oneyear[['year', 'month', 'day', 'hour','minute']])
    oneyear.drop(['month','day','hour','year','minute'], inplace = True,axis = 1)
    
    
    numgreater = (oneyear['WS_elevation_m'] <= elevation).sum()
    numtotal = oneyear.WS_elevation_m.size
    
    return float(numgreater)/float(numtotal)

#Get recovery values for each section and apply identifying information (well,core,section)
sr_path = root_path + 'IBP Project\Documents\Data Logs from Field Work\\'
sect_rec = pd.read_table(sr_path+'section_recovery.csv',sep=',',index_col=False)
#Get core number and section number from nominal bottom depth
sect_rec['core'] = np.floor(sect_rec.depth_bottom_cm/31+1)
sect_rec['section'] = sect_rec.depth_bottom_cm/10
sect_rec = sect_rec.drop('depth_bottom_cm',1)
sect_rec = sect_rec[['well','core','section','section_recovery_cm']]

#Calculate multiplier for each core
#This assumes all sections in a core compact evenly along the depth of the core
sr_mult = sect_rec.drop('section',1).groupby(['well','core'],as_index = False).sum()
#factor = (30 cm)/(Sum of all section lengths per core)
sr_mult['factor'] = (30./sr_mult.section_recovery_cm)
sr_mult.columns=['well','core','core_recovery_cm','factor']

#Calculate "stretched" length of each section
#section_length_cm =  section_recovery_cm*factor
sect_rec = pd.merge(sect_rec,sr_mult)
sect_rec['section_length_cm'] = sect_rec.section_recovery_cm*sect_rec.factor
sect_rec = sect_rec.drop(['core_recovery_cm'],1)

#Calculate cumulative sum of lengths per well (final value for each well should be 120 cm, as none hit refusal)
#Groups into new data frames by core, using the section as the index (dropping the "core" identifier as this is contained in the well and section)
#Calculates cumulative sum of all values down the index of each group
#Combines groups into a single data frame with well and section
cd = sect_rec.groupby(by=['well','section']).sum().groupby(level=[0]).cumsum().drop(['core','factor','section_recovery_cm'],1).reset_index()
cd.columns=['well','section','cum_depth_cm']
#Combines the two data frames using the common columns, 'well' and 'section', to match values to appropriate samples
sect_rec = pd.merge(sect_rec,cd)

#Get ground elevation from sensor metadata file
gr_el = sensor_meta[['data_id','ground_elev_ft']].drop_duplicates(keep = 'first').copy()
gr_el['ground_elev_m'] = gr_el.ground_elev_ft*0.3048
gr_el = gr_el.drop('ground_elev_ft',1)
gr_el.columns=['well','ground_elev_m']
#Combines the two data frames using the common column, 'well', to match values to appropriate samples (ground elevation is the same for all samples in a well, so results in duplicate rows in this column)
sect_rec = pd.merge(sect_rec,gr_el)
#Calculate elevation of top and bottom of each section and add as columns to main file
sect_rec['bottom_elev_m'] = sect_rec.ground_elev_m-sect_rec.cum_depth_cm/100.
sect_rec['top_elev_m'] = sect_rec.bottom_elev_m+sect_rec.section_length_cm/100.
#Drop ground elevation column now we're done with it
sect_rec = sect_rec.drop('ground_elev_m',1)

#sect_rec.to_csv("20180730_depth_info.csv",index = None)