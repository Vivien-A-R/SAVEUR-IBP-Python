# -*- coding: utf-8 -*-
"""
Function to get sensor metadata

Created on Mon Aug 14 15:54:06 2017

@author: Vivien
"""
import pandas as pd

def meta_get(data_path):
    sensor_meta = pd.read_table(data_path+'wl_position_meta.csv',
                                sep=',', index_col=False,parse_dates = ['date'])
    sensor_meta.columns = ['data_id', 'top_elev_ft', 'cable_length_ft',
                           'ground_elev_ft', 'lat', 'long','date_change']
    return sensor_meta

