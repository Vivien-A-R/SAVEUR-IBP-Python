# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 14:20:36 2017

@author: Vivien
"""
import pandas as pd
import numpy as np
import re

#data_path = 'D:\Dropbox\IBP_Python\scripts_and_filenames\\'
data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

#Change this to point to the boring log (short/fat "unmelted" format)
df_soil_column = pd.read_table(data_path+'Soil Characterization\\boring_log.csv',sep = ',')
wells = [s for s in list(df_soil_column) if "WLW" in s]

df_twist = pd.read_table(data_path+'Soil Characterization\\boring_log_twist.csv',sep = ',')
bores = df_twist.iloc[:,[1,2,3]].columns.tolist()

def soil_codes(x):  # For method 1
    return {
        'CL': '1',
        'CH': '2',
        'SP': '3',
        'ML': '4',
        'OL': '5',
        'Sandy CL' : '6'
        }[x]

def color_codes(x):  #For method 3
    return {
        'Brown':  '1',
        'Black':  '2',
        'Tan':    '3',
        'Gray':   '4',
        'Orange': '5'
        }[x]

#Run this function by itself to see the key/values for these.
def uct2num(): #Unique color/type to numeric (Generates codes for method 4)
    uniques = np.unique(df_soil_column[wells])
    unique_codes = {}
    i = 1
    for soil in uniques:
        unique_codes[soil] = i
        i = i+1
    return unique_codes

def unique_codes(x): return uct2num()[x]  #For method 4

pat = re.compile(r'(^\w+) (\D*)')  #Separates the one-word color from the rest of the string

#Replace individual values
def gc_byval(soilstring,method,verbose = True):
    methods = [1,2,3,4]
    #Takes an individual value and returns the code using the desired method
    if(method not in methods):
        print "WHOA, NELLY. There are only four methods!"
        result = 0
    elif(method is 1):
        if(verbose == True): print "Method 1: Replace field with numeric value for full soil code (text for number)"
        result = int(pat.sub(lambda m: soil_codes(m.group(2)),soilstring))
    elif(method is 2):
        if(verbose == True): print "Method 2: Replace field with color name (text for text)"
        result = pat.sub(lambda m: m.group(1),soilstring)
    elif(method is 3):
        if(verbose == True): print "Method 3: Replace field with numeric value for color name (text for number)"
        result = int(pat.sub(lambda m: color_codes(m.group(1)),soilstring))
    elif(method is 4):
        if(verbose == True): print "Method 4: Replace field with unique numeric value for soil color + type"
        result = unique_codes(soilstring)
    return result

#Generate a dataframe using chosen method (above)
def gcf(method): #Generate coded DataFrames from original; preserves shape (short/fat)
    df_codes = df_soil_column.copy()
    for well in wells:
        df_codes[well] = df_codes.apply(lambda x: gc_byval(x[well],method,False),axis = 1)
    return df_codes