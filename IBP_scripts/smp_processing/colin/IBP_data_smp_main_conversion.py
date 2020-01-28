# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:59:54 2017

@author: colinbphillips
"""
import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import os as os 



#main_data_path = "D:\Dropbox\PyPractice\smp_processing\colin"
main_data_path = "C:\Users\Vivien\Dropbox\IBP_Python\smp_processing\colin"
#processed_data_path = "D:\Dropbox\PyPractice\smp_processing\colin"
processed_data_path = "C:\Users\Vivien\Dropbox\IBP_Python\smp_processing\colin"


os.chdir(main_data_path)
ndatafile=pd.read_table('data_file_names_SMP.txt')##loads a text file with the file name of the dataset of interest.
dataname1=ndatafile.ix[0,0] ##gets file name from data file name txt file.

os.chdir(main_data_path)
data_id=os.path.join('text files',dataname1[0:4]) ##finds name of soil sensor (1 or 2)
file_path=os.path.join(main_data_path,data_id)


os.chdir(file_path)
#df_sensor_depth=pd.read_table('sensor_depths.txt',sep=',',index_col=False)
##### Note on data units - soil moisture data has raw and moisture. Raw is a frequency output (raw count). Moisture is in mm/100 mm. Standard unit is typically mm/m.
df_smp=pd.read_csv(dataname1,sep=',',skiprows=9,names=['index_drop','date_time','A1_raw','A1_moisture','A2_raw','A2_moisture','A3_raw','A3_moisture','A4_raw','A4_moisture','A5_raw','A5_moisture','A6_raw','A6_moisture','na_empty'],na_values=['INVALID          '],parse_dates=['date_time'])

df_smp.drop(['index_drop','na_empty'],1,inplace=True)

file_save_name=dataname1[0:4]+'_ibp_main.csv' ##creating filename based on the smp # to use to save data.
df_smp.to_csv(file_save_name,sep=',') ##saves processed data to new file