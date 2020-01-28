# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:44:02 2016

@author: colinbphillips
"""
####Looks like I will need to remove the quotes from the file. Their presence seems to cause havoc on the dtype of the data. Alternatively, forcing every dtype to change its type after loading in the data is an option (will add more lines of code as each line needs to be treated separately).

smp='SMP2'
import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import os as os #d
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

#main_data_path = "D:\Dropbox\PyPractice\smp_processing\colin"
main_data_path = "C:\Users\Vivien\Dropbox\IBP_Python\smp_processing\colin"
data_id=os.path.join('text files',smp)
file_path=os.path.join(main_data_path,data_id)
os.chdir(main_data_path)

#ndatafile=pd.read_table('data_file_names_SMP.txt')##loads a text file with the file name of the dataset of interest.
#dataname1=ndatafile.ix[0,0] ##gets file name from data file name txt file.
dataname1=smp+'_ibp_main.csv'


os.chdir(file_path)
#df_sensor_depth=pd.read_table('sensor_depths.txt',sep=',',index_col=False)
#dtypes1={'index_drop':int,'date_time':str, 'A1_raw':float,'A1_moisture':float,'A2_raw':float,'A2_moisture':float,'A3_raw':float,'A3_moisture':float,'A4_raw':float,'A4_moisture':float,'A5_raw':float,'A5_moisture':float,'A6_raw':float,'A6_moisture':float,'na_empty':object}
##### Note on data units - soil moisture data has raw and moisture. Raw is a frequency output (raw count). Moisture is in mm/100 mm. Standard unit is typically mm/m.
df_smp = pd.read_csv(dataname1,sep=',',index_col=0,parse_dates=['date_time'])
start = datetime.datetime.strptime('2016-07-10 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.datetime.strptime('2016-07-30 00:00:00', '%Y-%m-%d %H:%M:%S')
df_smp = df_smp[df_smp.date_time < end]
df_smp = df_smp[df_smp.date_time > start]


precip = pd.read_csv('C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\data_products\precip.csv',sep = ',')
precip.columns = ['date_time', 'precip_30min_in']
precip.date_time = pd.to_datetime(precip.date_time)

yticks = (10,20,30,40)
elmin, elmax = 5,45
texty = 30
textx = '2016-07-09 00:00:00'

f, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex = True,figsize = (12,6)) ##defining the figure
ax1.plot(df_smp.date_time,df_smp.A1_moisture,'-',color = 'black')
#ax1.text(textx,texty, "Sensor depth: 10 cm")
ax1.set_yticks(yticks)
ax1.set_ylim(elmin,elmax)
ax2.plot(df_smp.date_time,df_smp.A2_moisture,'-',color = 'black')
#ax2.text(textx,texty, "Sensor depth: 20 cm")
ax2.set_yticks(yticks)
ax2.set_ylim(elmin,elmax)
ax3.plot(df_smp.date_time,df_smp.A3_moisture,'-',color = 'black')
#ax3.text(textx,texty, "Sensor depth: 30 cm")
ax3.set_yticks(yticks)
ax3.set_ylim(elmin,elmax)

ax4.plot(precip.date_time, precip.precip_30min_in)
ax4.set_yticks((0,0.2,0.4))
#ax4.text(textx,0.5, "Precipitation (inches)",color='royalblue')
myFmt = mdates.DateFormatter('%B %d')
ax1.xaxis.set_major_formatter(myFmt)

#f.autofmt_xdate()
plt.subplots_adjust(hspace=0.0001)

####RAIN
precip = pd.read_csv('C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\data_products\precip.csv',sep = ',')
precip.columns = ['date_time', 'precip_30min_in']
precip.date_time = pd.to_datetime(precip.date_time)


#==============================================================================
# plt.figure(1)
# plt.subplot(6,2,1)
# plt.plot(df_smp.date_time,df_smp.A1_moisture,'-')
# plt.subplot(6,2,3)
# plt.plot(df_smp.date_time,df_smp.A2_moisture,'-')
# plt.subplot(6,2,5)
# plt.plot(df_smp.date_time,df_smp.A3_moisture,'-')
# plt.subplot(6,2,7)
# plt.plot(df_smp.date_time,df_smp.A4_moisture,'-')
# plt.subplot(6,2,9)
# plt.plot(df_smp.date_time,df_smp.A5_moisture,'-')
# plt.subplot(6,2,11)
# plt.plot(df_smp.date_time,df_smp.A6_moisture,'-')
#==============================================================================


#==============================================================================
# ymin, ymax = plt.subplot(6,2,1).get_ylim()
# plt.subplot(6,2,1).set_ylim(ymin,ymax)
# plt.subplot(6,2,3).set_ylim(ymin,ymax)
# plt.subplot(6,2,5).set_ylim(ymin,ymax)
# plt.subplot(6,2,7).set_ylim(ymin,ymax)
# plt.subplot(6,2,9).set_ylim(ymin,ymax)
# plt.subplot(6,2,11).set_ylim(ymin,ymax)
#==============================================================================


#==============================================================================
# 
# plt.subplot(6,2,2)
# plt.hist(df_smp.A1_moisture,bins=np.round(np.sqrt(len(df_smp.A1_moisture)),decimals = -1),range=[np.min(df_smp.A1_moisture),np.max(df_smp.A1_moisture)],normed=True)
# xmin, xmax = plt.subplot(6,2,2).get_xlim()
# plt.subplot(6,2,4)
# plt.hist(df_smp.A2_moisture,bins=np.round(np.sqrt(len(df_smp.A2_moisture)),decimals = -1),range=[np.min(df_smp.A2_moisture),np.max(df_smp.A2_moisture)],normed=True)
# plt.subplot(6,2,4).set_xlim(xmin,xmax)
# plt.subplot(6,2,6)
# plt.hist(df_smp.A3_moisture,bins=np.round(np.sqrt(len(df_smp.A3_moisture)),decimals = -1),range=[np.min(df_smp.A3_moisture),np.max(df_smp.A3_moisture)],normed=True)
# plt.subplot(6,2,6).set_xlim(xmin,xmax)
# plt.subplot(6,2,8)
# plt.hist(df_smp.A4_moisture,bins=np.round(np.sqrt(len(df_smp.A4_moisture)),decimals = -1),range=[np.min(df_smp.A4_moisture),np.max(df_smp.A4_moisture)],normed=True)
# plt.subplot(6,2,8).set_xlim(xmin,xmax)
# plt.subplot(6,2,10)
# plt.hist(df_smp.A5_moisture,bins=np.round(np.sqrt(len(df_smp.A5_moisture)),decimals = -1),range=[np.min(df_smp.A5_moisture),np.max(df_smp.A5_moisture)],normed=True)
# plt.subplot(6,2,10).set_xlim(xmin,xmax)
# plt.subplot(6,2,12)
# plt.hist(df_smp.A6_moisture,bins=np.round(np.sqrt(len(df_smp.A6_moisture)),decimals = -1),range=[np.min(df_smp.A6_moisture),np.max(df_smp.A6_moisture)],normed=True)
# plt.subplot(6,2,12).set_xlim(xmin,xmax)
#==============================================================================


#plt.figure(2)



























