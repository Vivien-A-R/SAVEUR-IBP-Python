# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:08:14 2017

@author: colinbphillips
"""
####This short script opens the SMP text file data and creates a text file in an easy to read format for data analysis. This script is intended to create the initial file that we will add new data to in the future.


import pandas as pd #pythons data/timeseries package
import os as os 


main_data_path = "D:\Dropbox\PyPractice\smp_processing\colin"
processed_data_path = "D:\Dropbox\PyPractice\smp_processing\colin"
os.chdir(main_data_path)

ndatafile=pd.read_table('data_file_names_SMP_process.txt')##loads a text file with the file name of the dataset of interest. The datafiles within this text file can be in any order and can mix and match SMP1 and SMP2. The script will use the number in the file name following SMP to identify where the data goes, and uses the date_time column to keep data in chronological order

for n in range(0,len(ndatafile)): ##This 'for loop' processes each sensor file given in the input text file.
    dataname1=ndatafile.ix[n,0] ##gets file name from data file name txt file.
    data_id=os.path.join('text files',dataname1[0:4]) ##finds name of soil sensor (1 or 2)
    file_path=os.path.join(main_data_path,data_id) ##selects the folder to access the data in for SMP1 or 2 depending on the data to be added.
    os.chdir(file_path) ##redirects to the data folder   
    main_file_dataname=dataname1[0:4]+'_ibp_main.csv' #identifies main data file to match the data that we are processing.
    df_main=pd.read_csv(main_file_dataname,sep=',',index_col=0,parse_dates=['date_time'])

    #df_sensor_depth=pd.read_table('sensor_depths.txt',sep=',',index_col=False)
    ##### Note on data units - soil moisture data has raw and moisture. Raw is a frequency output (raw count). Moisture is in mm/100 mm. Standard unit is typically mm/m.
    df_smp=pd.read_csv(dataname1,sep=',',skiprows=9,names=['index_drop','date_time','A1_raw','A1_moisture','A2_raw','A2_moisture','A3_raw','A3_moisture','A4_raw','A4_moisture','A5_raw','A5_moisture','A6_raw','A6_moisture','na_empty'],na_values=['INVALID          '],parse_dates=['date_time'])
    df_smp.drop(['index_drop','na_empty'],1,inplace=True)

    df_smp_all=df_main.append(df_smp, ignore_index=True) #adding the new data to the end of the old data. In the following lines we will reorder the data into chronological order, reset the index, and correct the run_time column.
    df_smp_all.sort_values('date_time',inplace=True) #sorts data by date and time, will only change data if the files to process were listed out in chronological order.     
    df_smp_all.drop_duplicates(subset='date_time',keep='first',inplace=True)  #removes data with duplicted date time values.  
    df_smp_all.reset_index(inplace=True,drop=True) #will only change index if the data were not added in chronological order.
    df_smp_all.to_csv(main_file_dataname,sep=',') ##saves processed data to new file, overwriting previous main data file.