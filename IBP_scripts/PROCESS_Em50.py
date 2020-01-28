# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 14:39:24 2018

@author: Vivien

Data processing for Decagon/METER EC sensors
"""

import pandas as pd
import os as os

raw_data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Data from Field Tablet\Em50 Loggers\Text files\\'
processed_data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Deprecated Data Folder\Electrical Conductivity\\'

filenames = os.listdir(raw_data_path)
prefix = "ECL"
suffix = ".txt"
af = [filename for filename in filenames if (filename.startswith(prefix) and filename.endswith(suffix))]

if(os.path.isfile(processed_data_path + 'ECL_datafiles.csv')==False):
    df_empty = pd.DataFrame({'files' : []})
    print "NO DATAFILE PROCESSING HISTORY FOUND: Creating datafiles.csv"
    df_empty.to_csv(processed_data_path + 'ECL_datafiles.csv',sep=',',columns = ['files'],index = False) ##saves new processed data file.

processed_files = pd.read_csv(processed_data_path+'ECL_datafiles.csv',names = ['files'])
sensor_files = pd.DataFrame(af,columns = ['files'])
new_files = sensor_files[~sensor_files.files.isin(processed_files.files)]

for f in new_files.files:
    
    p = raw_data_path + f
    print f
    t = pd.read_table(p,sep = '\t')
    if(len(t.columns) == 9):
        colnames = ['date_time','A_VWC_frac','A_Temp_C','A_EC_mS/cm',
                    'B_VWC_frac','B_Temp_C','B_EC_mS/cm',
                    'w_EC_mS/cm','w_Temp_C']
    else:
        colnames = ['date_time','A_VWC_frac','A_Temp_C','A_EC_mS/cm',
                    'B_VWC_frac','B_Temp_C','B_EC_mS/cm']
    t.columns = colnames
    t.date_time = pd.to_datetime(t.date_time)
    
    
    id = f[0:4]
    
    #Convert strings to numerics or NaNs as appropriate
    mvs = t.columns[1:]
    t[mvs] = t[mvs].apply(pd.to_numeric,errors = 'coerce')
    
    #t['date_time'] = t['date_time'].dt.round('30min') #Rounds for clock lag (should never be more than a few minutes)
    
    #Create main file if not already exisitng
    if(os.path.isfile(processed_data_path + id + '_ibp_main.csv') == False):
        missing = id
        print("NO FILE FOUND FOR " + missing + ": Creating " + missing + "_ibp_main.csv")
        if(len(t.columns) == 9):
            df_empty = pd.DataFrame({'date_time':[],
                                     'A_VWC_frac':[],
                                     'A_Temp_C':[],
                                     'A_EC_mS/cm':[],
                                     'B_VWC_frac':[],
                                     'B_Temp_C':[],
                                     'B_EC_mS/cm':[],
                                     'w_EC_mS/cm':[],
                                     'w_Temp_C':[]
                                     })
        else:
            df_empty = pd.DataFrame({'date_time':[],
                                     'A_VWC_frac':[],
                                     'A_Temp_C':[],
                                     'A_EC_mS/cm':[],
                                     'B_VWC_frac':[],
                                     'B_Temp_C':[],
                                     'B_EC_mS/cm':[]
                                     })
        
        df_empty.to_csv(processed_data_path + missing + '_ibp_main.csv',sep=',',index = None)
    
    #Get preexisting file (possibly newly-created in prev. line) and append
    m = pd.read_csv(processed_data_path + id + '_ibp_main.csv',parse_dates = ['date_time'])
    m = m.append(t,ignore_index = True)
    
    if(len(m.columns) == 9):
        m = m[['date_time',
              'A_VWC_frac','A_Temp_C','A_EC_mS/cm',
              'B_VWC_frac','B_Temp_C','B_EC_mS/cm',
              'w_EC_mS/cm','w_Temp_C']]
    else:
        m = m[['date_time',
              'A_VWC_frac','A_Temp_C','A_EC_mS/cm',
              'B_VWC_frac','B_Temp_C','B_EC_mS/cm']]
        
    
    #Cleanup (remove overlap, sort chronologically)
    m.drop_duplicates(subset='date_time',keep='first',inplace=True)  #removes data with duplicted date time values.
    m.sort_values('date_time',inplace=True) #sorts data by date and time, will only change data if the files to process were listed out in chronological order.
    m.reset_index(inplace=True,drop=True) #will only change index if the data were not added in chronological order.
    
    # =============================================================================
    # #Sets up a continuous 30-minute time series so we can fill gaps with NaN values
    # #Has to go here because there are gaps between files in some cases
    # tstart = m.date_time.iloc[0]
    # tend = m.date_time.iloc[-1]
    # tind = pd.DataFrame(pd.date_range(tstart,tend,freq = '30min'))
    # tind.columns = ['date_time']
    # m = pd.merge(tind,m,'left')
    # =============================================================================
    
    m.to_csv(processed_data_path + id + '_ibp_main.csv',sep=',',index = None,float_format='%10.6f')

processed_files = processed_files.append(new_files,ignore_index = True)
processed_files.to_csv(processed_data_path + 'ECL_datafiles.csv',sep=',',index = False)