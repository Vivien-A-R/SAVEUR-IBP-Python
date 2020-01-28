# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:56:04 2019

@author: Packman-Field
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Deprecated Data Folder\Data From SMP\\'

probes = ['1','2']
for p in probes:
    fname = 'SMP'+p+'_ibp_main.csv'
    df = pd.read_csv(data_path+fname,parse_dates = ['date_time'])
    depths = [10,20,40,60,80,100]
    times = df['date_time']
    timestrs = times.dt.strftime('%Y-%m-%d %H:%M')
    mcols = ['a1_moisture','a2_moisture','a3_moisture','a4_moisture','a5_moisture','a6_moisture']
    df_mod = df[mcols].transpose()
    df_mod.index = depths
    df_mod.columns = timestrs
    
    df_mod.loc[30] = 100
    df_mod.loc[50] = 100
    df_mod.loc[70] = 100
    df_mod.loc[90] = 100
    df_mod.sort_index(axis = 0,inplace = True)
    
    mask = df_mod.isnull()
    df_mod.replace(100,np.nan,inplace=True)
    df_mod.interpolate(method='index',inplace = True)
    
    plt.figure()
    sns.heatmap(df_mod, mask = mask,vmin = 0, vmax = 60,cmap="YlGnBu")
    plt.tight_layout()