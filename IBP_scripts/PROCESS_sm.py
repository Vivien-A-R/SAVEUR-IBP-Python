# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:32:59 2017

@author: Vivien
"""
import pandas as pd #pythons data/timeseries package
import os as os #data in/out package

###############################################################################
#raw_data_path = 'D:\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Data from Field Tablet\SMP Data\\text files\\'
#processed_data_path = 'D:\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Data From SMP\\'
raw_data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Data from Field Tablet\SMP Data\\text files\\'
processed_data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Data From SMP\\'
###############################################################################

filenames = os.listdir(raw_data_path)
prefix = "SM"
suffix = ".txt"
af = [filename for filename in filenames if (filename.startswith(prefix) and filename.endswith(suffix))]

def header_count(file_path):
    i=-1
    data_start_flag='Index'
    temp_line=('placeholder')
    header_bottom = 0
    #Change January 2019: 'rU' to treat carriage return \r as newline \n; fix miscounting error
    temp_file=open(file_path,'rU') ##opens the file without loading in the data, remember to close the file once done using it.
    while (temp_line.find(data_start_flag)==i):
        header_bottom = header_bottom + 1
        temp_line = temp_file.readline()
        #print(str(header_bottom) + ": " + temp_line)
    temp_file.close()
    return header_bottom

if(os.path.isfile(processed_data_path + 'datafiles.csv')==False):
    df_empty = pd.DataFrame({'files' : []})
    print "NO DATAFILE PROCESSING HISTORY FOUND: Creating datafiles.csv"
    df_empty.to_csv(processed_data_path + 'datafiles.csv',sep=',',columns = ['files'],index = False) ##saves new processed data file.

processed_files = pd.read_csv(processed_data_path+'datafiles.csv',names = ['files'])
sensor_files = pd.DataFrame(af,columns = ['files'])
new_files = sensor_files[~sensor_files.files.isin(processed_files.files)]

for f in new_files.files:
    p = raw_data_path + f
    print f
    c = header_count(p)
    t = pd.read_table(p,sep=',',index_col=False,skiprows=c,
                      names = ['index','date_time','a1_raw','a1_moisture','a2_raw','a2_moisture','a3_raw','a3_moisture','a4_raw','a4_moisture','a5_raw','a5_moisture','a6_raw','a6_moisture','null'],parse_dates=['date_time'])
    t = t.drop(['index','null','a1_raw','a2_raw','a3_raw','a4_raw','a5_raw','a6_raw'],1)
    t['qual_c'] = 1
    id = f[0:4]

    #Convert strings to numerics or NaNs as appropriate
    mvs = t.columns[1:7]
    t[mvs] = t[mvs].apply(pd.to_numeric,errors = 'coerce')
    t['date_time'] = pd.to_datetime(t['date_time'])

    t['date_time'] = t['date_time'].dt.round('30min') #Rounds for clock lag (should never be more than a few minutes)

    #Create main file if not already exisitng
    if(os.path.isfile(processed_data_path + id + '_ibp_main.csv')==False):
        missing = id
        print("NO FILE FOUND FOR " + missing + ": Creating " + missing + "_ibp_main.csv")
        df_empty = pd.DataFrame({'date_time':[],
                                 'a1_moisture':[],
                                 'a2_moisture':[],
                                 'a3_moisture':[],
                                 'a4_moisture':[],
                                 'a5_moisture':[],
                                 'a6_moisture':[],
                                 'qual_c':[]
                                })
        df_empty.to_csv(processed_data_path + missing + '_ibp_main.csv',sep=',',index = None)

    #Get preexisting file (possibly newly-created in prev. line) and append
    m = pd.read_csv(processed_data_path + id + '_ibp_main.csv',parse_dates = ['date_time'])
    m = m.append(t,ignore_index = True)

    m = m[['date_time', #Reorder columns
           'a1_moisture',
           'a2_moisture',
           'a3_moisture',
           'a4_moisture',
           'a5_moisture',
           'a6_moisture',
           'qual_c']]
    #Cleanup (remove overlap, sort chronologically)
    m.drop_duplicates(subset='date_time',keep='first',inplace=True)  #removes data with duplicted date time values.
    m.sort_values('date_time',inplace=True) #sorts data by date and time, will only change data if the files to process were listed out in chronological order.
    m.reset_index(inplace=True,drop=True) #will only change index if the data were not added in chronological order.

    #Sets up a continuous 30-minute time series so we can fill gaps with NaN values
    #Has to go here because there are gaps between files in some cases
    tstart = m.date_time.iloc[0]
    tend = m.date_time.iloc[-1]
    tind = pd.DataFrame(pd.date_range(tstart,tend,freq = '30min'))
    tind.columns = ['date_time']
    m = pd.merge(tind,m,'left')

    #Where there are gaps, set qc flag
    m.loc[m[m.isnull().any(1)].index,'qual_c'] = 0
    #print(len(m[m[mvs].isnull().any(1)]) - len(m[m[mvs].isnull().all(1)])) #How many rows are partially incomplete? (Only some values are na)
    m.to_csv(processed_data_path + id + '_ibp_main.csv',sep=',',index = None,float_format='%10.6f')

processed_files = processed_files.append(new_files,ignore_index = True)
processed_files.to_csv(processed_data_path + 'datafiles.csv',sep=',',index = False)

#Data load and plotting, for checking
#mp1 = pd.read_csv(processed_data_path + 'SMP1' + '_ibp_main.csv',parse_dates = ['date_time'])
#mp2 = pd.read_csv(processed_data_path + 'SMP2' + '_ibp_main.csv',parse_dates = ['date_time'])
#mp1.plot(x = 'date_time',y = mvs,title = 'SMP1')
#mp2.plot(x = 'date_time',y = mvs,title = 'SMP2')
