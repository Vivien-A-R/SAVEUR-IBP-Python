# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 14:39:18 2017

@author: Vivien
"""
import pandas as pd
import re
from meta_get import meta_get


#data_path = 'D:\Dropbox\IBP_Python\scripts_and_filenames\\'
data_path = 'C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\'

#Boring logs from corer (1-inch incr) and twist auger (10-cm incr)
df_soil_column = pd.read_table(data_path+'raw_nt_data\\boring_log.csv', sep=',')
df_soil_ctwist = pd.read_table(data_path+'raw_nt_data\\boring_log_twist.csv', sep=',')

#Depth info calculated from core/section recovery
depth_elev = pd.read_table(data_path+'raw_nt_data\\20171025_depth_info.csv', sep=',')

#Sensor metadata
sensor_meta = meta_get(data_path)[['data_id', 'ground_elev_ft']]
sensor_meta.columns = ['well', 'ground_elev_ft']

# Melt (turn each well column in the short, fat frame into a three-column chunk (depth, well, value) of a long, skinny frame
wells = [s for s in df_soil_column if "WLW" in s]
wells.remove('WLW11')
soil_melted = pd.melt(df_soil_column, id_vars='depth_cm', value_vars=wells)
soil_melted.columns = ['depth_cm', 'well', 'type_color']

#Returns a df of inch-thick slices from the soil df (should not be very many)
def slicer(well_id,bottom_depth,top_depth):
    s_slices = soil_melted[soil_melted['well'] == well_id]
    s_slices = s_slices[s_slices['depth_cm'] <= bottom_depth]
    s_slices = s_slices[s_slices['depth_cm'] >= top_depth]
    return s_slices

#Assigns a soil type to an irregularly-shaped section based on the boring logs
def choose_type(well_id,bottom_depth,top_depth):
    slices = slicer(well_id,bottom_depth,top_depth) #Get a stack of slices
    stype = slices['type_color'].mode() #Get the most common soil types occuring within those slices (CAN BE PLURAL)
    #print(well_id + ' ' + str(bottom_depth) + ' ' + str(top_depth) + ' ' + str(len(slices)) + ' ' + str(len(stype)))

    if(len(slices) == 1): return slices['type_color'].item() #i.e. there is only one item in the stack, choose that one
    elif(len(stype) == 1): return stype.item() #i.e. there is only one most-common type in the stack (no ties), choose that one         
    else: # In case of tie, slowly expand the area we're looking at by 1cm at a time until the tie is broken
        count,b,t,q = 0,bottom_depth,top_depth,len(stype)
        while(q > 1 and count < 40):
            b = b - 0.1
            t = t + 0.1
            temp = slicer(well_id,b,t)
            stype = temp['type_color'].mode()
            q = len(stype)
            count = count + 1
        return stype.item()

#Info from subsections, irregularly spaced (use choose_type() assign types to these sections)
sections = depth_elev[['well','section','section_length_cm','cum_depth_cm']].copy()
sections['top_depth_cm'] = sections.cum_depth_cm-sections.section_length_cm
sections = sections[sections.section_length_cm != 0]
sections['type'] = sections.apply(lambda row: choose_type(row['well'],row['cum_depth_cm'],row['top_depth_cm']),axis = 1)


soil_irr = sections[['well','cum_depth_cm','type','section']]

soil_mtwist = pd.melt(df_soil_ctwist,id_vars = 'depth_cm')
soil_mtwist.columns = ['cum_depth_cm', 'well', 'type']
soil_mtwist['section'] = soil_mtwist.cum_depth_cm/10
soil_irr = soil_irr.append(soil_mtwist).reset_index(drop = True)

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
    uniques = soil_melted.type_color.unique()
    uniques.sort()
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
def gcf_melted(method):
    df_codes = soil_irr.copy()
    df_codes['type'] = df_codes.apply(lambda x: gc_byval(x['type'],method,False),axis = 1)
    return df_codes[['well','section','cum_depth_cm','type']]
