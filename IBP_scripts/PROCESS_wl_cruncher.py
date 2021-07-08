# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:10:45 2017
Last updated Aug 2, 2017

@author: Vivien; from Colin's processing scripts
"""
import numpy as np
import pandas as pd
import os as os
from meta_get import meta_get
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# qc values:
# 0 for non-physical data (sensor out of well, ice encased, etc.)
# -1 for data that still carries information (sensor in place and functional, but dry)

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 50)

##############################################################################
raw_data_path = "C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Data from Field Tablet\WinSitu\Exported Data\\"
data_path = processed_data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'
##############################################################################

sensor_meta = meta_get(data_path)

#List all filenames in a folder that end with the chosen suffix (.csv)
def find_csv_filenames(path_to_dir ,prefix = "",suffix=""):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if (filename.startswith(prefix) and filename.endswith(suffix))]

def find_new_files(sensor="BARO"):
    if(os.path.isfile(processed_data_path + 'datafiles.csv')==False):
        df_empty = pd.DataFrame({'files' : []})
        print "NO DATAFILE PROCESSING HISTORY FOUND: Creating datafiles.csv"
        df_empty.to_csv(processed_data_path + 'datafiles.csv',sep=',',columns = ['files'],index = False) ##saves new processed data file.

    all_files = pd.read_csv(processed_data_path+'datafiles.csv',names = ['files'])
    sensor_files = pd.DataFrame(find_csv_filenames(raw_data_path + sensor, suffix = '.csv'),columns = ['files'])
    new_files = sensor_files[~sensor_files.files.isin(all_files.files)] #Find all baro files that are not in all_files

    print("Found "+ str(len(new_files))+ " new file(s) in " + sensor)
    return new_files

def get_rloc(baro = False):
    #Get all the directories in "Exported Data" containing LevelTroll Data (not Baro or Firmware)
    d = raw_data_path
    rloc = pd.DataFrame([[o,os.path.join(d,o)] for o in os.listdir(d) if os.path.isdir(os.path.join(d,o)) if o[0:2] == 'WL'],
                                 columns = ['well_id','raw_path'])
    rloc = rloc[rloc.well_id.str.contains("WLW11") == False].reset_index(drop = True)
    #raw_locations['well_id_2'] = raw_locations.well_id.str.split().str.get(0)
    return rloc
    
# Find any duplicate folders (WLW1 and WLW1 (1) for example)
# resulting from syncing errors across two field tablets and combine their data
#def combine_folders():
#    raw_locations = get_rloc(True)
#    if raw_locations.well_id.str.split().str.get(1) != '(0)':
#        print "WOO."

#find number of "header" lines to skip by stepping through with a counter until the flag ("Date and Time") string is found
def header_count(file_path):
    i=-1
    data_start_flag='Date and Time,Seconds'
    temp_line=('placeholder')
    header_bottom = 0
    temp_file=open(file_path,'r') ##opens the file without loading in the data, remember to close the file once done using it.
    while (temp_line.find(data_start_flag)==i):
        temp_line=temp_file.readline()
        header_bottom = header_bottom + 1
    temp_file.close()
    return header_bottom

#Retreive correct value from metadata file based on date
def get_cv(date_in,df_elev):
    len_df = df_elev[['cable_length_ft','date_change']]
    val = len_df[len_df['date_change'] < date_in].iloc[-1]['cable_length_ft']
    return val

#Make a new (empty) baro file
def baro_create():
    print "NO BARO FILE FOUND: Creating baro_ibp_main.csv"
    df_empty = pd.DataFrame({'date_time':[],
                             'run_time':[],
                             'pressure_pa':[],
                             'temperature_c':[],
                             'qual_c':[]
                             })

    df_empty.to_csv(processed_data_path + 'baro_ibp_main.csv',sep=',',index = False)

def baro_populate():
    if(os.path.isfile(processed_data_path + 'baro_ibp_main.csv')==False): baro_create()

    new_files = find_new_files("BARO")
    if(len(new_files)!=0):
        for newfile in new_files.files:
            print newfile
            file_path = raw_data_path + "BARO\\" + newfile
            header_rows = header_count(file_path)

            df_baro=pd.read_table(file_path,sep=',',index_col=False,skiprows=header_rows,
                                  names=['date_time','run_time','pressure_mmHg','temperature_c'],
                                  parse_dates=['date_time'],engine='python')

            #Process new data
            df_baro['pressure_pa']=df_baro.pressure_mmHg*133.322 #convert from mmHg to Pa
            df_baro['qual_c']=np.ones((len(df_baro),1))
            df_baro.drop(['pressure_mmHg'],1,inplace=True)
            df_baro['qual_c'].iloc[-1] = 0

            #Merge new data with old data
            df_baro_main=pd.read_table(processed_data_path+'baro_ibp_main.csv',sep=',',parse_dates=['date_time'])##opening datafile to append new data to.
            df_baro_all = df_baro_main.append(df_baro, ignore_index=True)
            df_baro_all.drop_duplicates(subset='date_time',keep='first',inplace=True)## removes duplicate date_times from the data, this way you can add in a file that has past data and it will only add the new data to the main file.
            df_baro_all.sort_values('date_time',inplace=True) #sorts data by date and time, will only change data if the files to process were listed not in chronological order.
            df_baro_all.reset_index(inplace=True,drop=True) #will only change index if the data were added not in chronological order.
            df_baro_all.dropna(how='any',inplace=True) ##removes any nan (not a number) values, one gets introduced to the end of the data as an extra blank line.

            #reorder columns
            df_baro_all = df_baro_all[['date_time','run_time','temperature_c','pressure_pa','qual_c']]

            #Save to csv
            df_baro_all.to_csv(processed_data_path+'baro_ibp_main.csv',sep=',',index = False)

        #Add the newly-added files to the datafiles.csv file, so that they are not reprocessed
        all_files = pd.read_csv(processed_data_path+'datafiles.csv',header = 0)
        all_files = all_files.append(new_files,ignore_index = True)
        all_files.to_csv(processed_data_path + 'datafiles.csv',sep=',',index = False)

def leveltroll_create(missing_well):
    print("NO FILE FOUND FOR " + missing_well + ": Creating " + missing_well + "_ibp_main.csv")
    df_empty = pd.DataFrame({'date_time':[],
                            'run_time':[],
                            'pressure_pa':[],
                            'temperature_c':[],
                            'sensor_elev_m':[],
                            'depth_m':[],
                            'WS_elevation_m':[],
                            'qual_c':[]
                            })

    df_empty.to_csv(processed_data_path + missing_well + '_ibp_main.csv',sep=',',index = False)

def leveltroll_populate():
    timesum = 0
    timecount = 0
    #Update baro files
    baro_populate()
    df_baro=pd.read_table(processed_data_path + 'baro_ibp_main.csv',sep=',',parse_dates=['date_time']) ##loads in the baro data
    df_baro.rename(columns={'pressure_pa':'pressure_baro','qual_c':'qc_baro'},inplace=True) #Renames pressure and qual_c to avoid collision
    df_baro_merge=df_baro.drop(['run_time','temperature_c'],1) #Removes unnecessary columns to avoid collision

    #Get all the directories in "Exported Data" containing LevelTroll Data (not Baro or Firmware)
    raw_locations = get_rloc(False)

    #Remove BaroMerge files (unneeded and can be replaced if necessary using tablet software)
    for directory in raw_locations.raw_path:
        t = find_csv_filenames(directory,suffix = 'BaroMerge.csv')
        for item in t: os.remove(os.path.join(directory,item))

    #Step through all wells and populate!
    #This layer is on a single-sensor basis; the processed files are not added to the list of processed files until the end of each loop (all of them have been processed)
    for well in raw_locations.well_id:
        if(os.path.isfile(processed_data_path + well + '_ibp_main.csv')==False): leveltroll_create(well)

        new_files = find_new_files(well)

        if(len(new_files)!=0):
            
            df_elev=sensor_meta[sensor_meta.data_id == well] ##selecting elevation data from file based on the site ID.

            #This layer is on a file-by-file basis with a different new raw file being processed per step
            for newfile in new_files.files:
                time1 = time.time()
                print newfile
                
                processed_file=well+'_ibp_main.csv' ##creating filename based on the site ID to use to save data.
                df_wl_main=pd.read_table(processed_data_path + processed_file,sep=',',parse_dates=['date_time'])##opening datafile to append new data to.

                file_path = raw_data_path + well + "\\" + newfile
                #header_rows = header_count(file_path) + len(df_wl_main) ## Skips everything already incorporated (do this more dynamically using date ranges) #Breaks the run_time column on WLW14??? Don't do this.
                header_rows = header_count(file_path) ## Skips everything already incorporated (do this more dynamically using date ranges)

                #Calculate depth of water above sensor from pressure data
                #the saved output data will only have as many lines as the baro data file because only the corrected data is kept.
                df_wl=pd.read_table(file_path,sep=',',index_col=False,skiprows=header_rows,names=['date_time','run_time','pressure_psi','temperature_c','depth'],parse_dates=['date_time'],engine='python')
                if(len(df_wl) > 0):
                    # This sensor had a calibration error and the datafiles cannot be modified before export, unfortunately
                    if(newfile in ['WLW14_2017-08-18_12-01-39-212.csv','WLW14_2017-09-07_13-18-15-389.csv']):
                        df_wl['pressure_psi'] = df_wl['pressure_psi'] + 14.4965
                        print "Corrected above file for pressure error!"
                    df_wl['pressure_pa']=df_wl.pressure_psi*6894.76 #convert psi to pa
                    
                    df_wl['qual_c']=np.ones((len(df_wl),1)) #adding quality control column, currently all data is considered fine. When adding new data we will mark that location as a 0 to indicate when the data was sampled.
                    df_wl_c=pd.merge(df_wl,df_baro_merge,on='date_time') # Combine data frames along date axis
                    df_wl_c['depth_m']=((df_wl_c.pressure_pa-df_wl_c.pressure_baro))/(1000*9.81) #Calculate depth using LevelTroll and BaroTroll pressure readings
                    df_wl_c.drop(['depth','pressure_baro','pressure_psi'],1,inplace=True) #Drop uncorrected depth; Drop baro pressure; Drop pressure in psi
                    
                    #Calculate elevation of water table
                    if len(df_elev.index) == 1:
                        df_wl_c['cable_length'] = df_elev.cable_length_ft.item()
                        df_wl_c['top_elev_ft'] = df_elev.top_elev_ft.item()
                    else:
                        df_wl_c['cable_length'] = df_wl_c.apply(lambda x: get_cv(x['date_time'],df_elev),axis = 1)
                        df_wl_c['top_elev_ft'] = df_elev['top_elev_ft'].drop_duplicates().item()
                    
                    df_wl_c['sensor_elev_m'] = (df_wl_c['top_elev_ft']-df_wl_c['cable_length'])*0.3048
                    df_wl_c['WS_elevation_m']=df_wl_c['sensor_elev_m']+df_wl_c.depth_m ##Water_elevation = Top hanger elevation - cable_length + water depth (all in meters)
                    df_wl_c['qual_c'] = df_wl_c['qc_baro'] * df_wl_c['qual_c'] #Lazy way of making sure that if qual_c is 0 for baro, it's 0 for data products as well.
                    df_wl_c.drop(['qc_baro','cable_length','top_elev_ft'],1,inplace = True)
                    noiseamp = 0.006 #Amplitude of noise about mean (meters); about 6mm noise
                    df_wl_c.loc[df_wl_c.depth_m <= 3*noiseamp,"qual_c"] = -1 #Set qual_c to -1 for measurements below the sensor
                    df_wl['qual_c'].iloc[-1] = 0 #set last value to 0
                    
                    #Merge with existing csv file
                    df_wl_all=df_wl_main.append(df_wl_c, ignore_index=True) #adding the new data to the end of the old data. In the following lines we will reorder the data into chronological order, reset the index, and correct the run_time column.
    
                    df_wl_all.drop_duplicates(subset='date_time',keep='first',inplace=True)  #removes data with duplicted run time values.
                    df_wl_all.sort_values('date_time',inplace=True) #sorts data by date and time, will only change data if the files to process were listed out in chronological order.
                    df_wl_all.reset_index(inplace=True,drop=True) #will only change index if the data were not added in chronological order.
                    df_wl_all.dropna(how='any',inplace=True)
    
                    df_wl_all = df_wl_all[['date_time','run_time','pressure_pa','temperature_c','sensor_elev_m','depth_m','WS_elevation_m','qual_c']]
                    df_wl_all.to_csv(processed_data_path + processed_file,sep=',',index = False) ##saves processed data to csv
                    
                    time2 = time.time()
                    timesum = timesum + time2-time1
                    timecount = timecount + 1

            #Add the newly-added files to the datafiles.csv file, so that they are not reprocessed (all new processed files at once)
            all_files = pd.read_csv(processed_data_path+'datafiles.csv',header = 0)
            all_files = all_files.append(new_files,ignore_index = True)
            all_files.to_csv(processed_data_path + 'datafiles.csv',sep=',',index = False)
    if(timecount > 0):print("Average time per file: " + str(timesum/timecount) + " ms")
    

def timesnip(sensor_id,start,end,show_suspect=True):
    #Lets you choose only a certain time span for a single sensor for testing purposes
    tpath = data_path + sensor_id + '_ibp_main.csv'
    datafile = pd.read_csv(tpath,parse_dates = ['date_time'])
    datestart = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M')
    if type(end) == str: dateend = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M')
    else: dateend = datestart + datetime.timedelta(hours = end)

    df_snip = datafile[(datafile['date_time'] > datestart) & (datafile['date_time'] < dateend) ]

    if(show_suspect == False): df_snip.loc[df_snip.qual_c < 1,"WS_elevation_m"]=np.nan #Skip already-flagged values
    fig,ax = plt.subplots(figsize = (12,4))
    ax.plot(df_snip.date_time,df_snip.WS_elevation_m,label = 'Water Level') #Shows the original signal
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %m %d %H%M' ))
    fig.autofmt_xdate()
    return df_snip

def qc_set(sensor_id,start,end,val=0,commit=True):
    #Lets you set a series of consecutive qual_c values for a single sensor
    tpath = data_path + sensor_id + '_ibp_main.csv'
    datafile = pd.read_csv(tpath,parse_dates = ["date_time"])
    datestart = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M')
    if type(end) == str: dateend = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M')
    else: dateend = datestart + datetime.timedelta(hours = end)

    inds = datafile[(datafile['date_time'] > datestart) & (datafile['date_time'] < dateend) ].index
    datafile.loc[inds,'qual_c'] = val

    df_chk = datafile.copy()
    df_chk.loc[df_chk.qual_c < 1,"WS_elevation_m"] = np.nan
    fig,ax = plt.subplots(figsize = (12,4))
    ax.plot(df_chk.date_time, df_chk.WS_elevation_m,label = 'Water Level') #Shows the original signal
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y %m %d %H%M'))
    fig.autofmt_xdate()

    if commit == True:
        #Lets you save that modified datafile as a csv
        print "Rows flagged:"
        print (datafile[datafile.qual_c < 1.0])
        file_save_name = data_path + sensor_id + '_ibp_main.csv'
        datafile.to_csv(file_save_name,sep=',',index = False) ##saves processed data to new file
        print file_save_name

#Trim down data to only useful info (decrease duplicates)
def data_pare():
    files = find_csv_filenames(processed_data_path, suffix=".csv",prefix = "WL" )
    for filen in files:
        pared_name = filen.split('_')[0] + '_GMP_simple.csv'
        df_wl_main=pd.read_table(processed_data_path + filen,sep=',',parse_dates=['date_time'])
        df_wl_pared = df_wl_main[['date_time','pressure_pa','temperature_c','WS_elevation_m','qual_c']]
        #df_wl_pared.to_csv(processed_data_path +'pared\\'+ pared_name,sep=',',index = False)
        print df_wl_pared.head()

#Smush it all into one file.
def data_mash():
    files = find_csv_filenames(processed_data_path, suffix=".csv" ,prefix = "WL")
    for each in [name + '_ibp_main.csv' for name in ['WLW11','WLW12','WLW13','WLW14']]: files.remove(each)
    df_wl_mashed = pd.DataFrame({'na':[]})
    for filen in files:
        well = filen.split('_')[0]
        df_wl_main=pd.read_table(processed_data_path + filen,sep=',',parse_dates=['date_time'])
        df_wl_pared = df_wl_main[['date_time','pressure_pa','temperature_c','WS_elevation_m','qual_c']]
        df_wl_pared.set_index('date_time',inplace = True)
        nameso = list(df_wl_pared)
        namesp = [well + "_" + col for col in nameso]
        df_wl_pared.columns = namesp
        df_wl_mashed = pd.concat([df_wl_mashed,df_wl_pared],axis = 1,join = 'outer')
        #print df_wl_mashed
    baro_pared = pd.read_table(processed_data_path + 'baro_ibp_main.csv',sep=',',parse_dates=['date_time'])
    baro_pared = baro_pared[['date_time','pressure_pa','temperature_c','qual_c']]
    baro_pared.set_index('date_time',inplace = True)
    nameso = list(baro_pared)
    namesp = ["BARO_" + col for col in nameso]
    baro_pared.columns = namesp
    df_wl_mashed = pd.concat([df_wl_mashed,baro_pared],axis = 1,join = 'outer')
    #df_wl_mashed = df_wl_mashed["2016-07-09 13:00:00":"2017-07-09 13:00:00"]
    df_wl_mashed = df_wl_mashed.dropna(axis = 1, how = "all")
    #df_wl_mashed.to_csv(processed_data_path +'pared\\GMP_20160707_20170707.csv',sep=',')
    return df_wl_mashed

def av_year():
    files = find_csv_filenames(data_path,"WL",".csv")
    for f in files:
        tpath = processed_data_path + f

        datafile = pd.read_csv(tpath,parse_dates=['date_time'])
        sdate = datafile.date_time.min()
        edate = datafile.date_time.max()

        start_str = str(sdate.year) + str(sdate.month).rjust(2,'0') + str(sdate.day).rjust(2,'0')
        end_str = str(edate.year) + str(edate.month).rjust(2,'0') + str(edate.day).rjust(2,'0')

        wl_series = datafile[['date_time','WS_elevation_m']]
        oneyear = wl_series.groupby([lambda x : wl_series['date_time'][x].month,
                                  lambda x: wl_series['date_time'][x].day,
                                  lambda x: wl_series['date_time'][x].hour,
                                  lambda x: wl_series['date_time'][x].minute]).mean()
        oneyear.reset_index(inplace = True)
        oneyear.columns = ['month','day','hour','minute','WS_elevation_m']
        oneyear['year']=1900
        oneyear['date_time']=pd.to_datetime(oneyear[['year', 'month', 'day', 'hour','minute']])

        average_year = oneyear[['date_time','WS_elevation_m']]

        titlestring = f.split('_')[0] + '_avg_yr_' + start_str + '_' + end_str + '.csv'
        avyear_folder = "C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\Water Level Derived Products\\Avyear\\"
        average_year.to_csv(avyear_folder + titlestring, index = None)
