# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:10:45 2018

@author: Vivien
"""

from __future__ import division
import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import os as os #data in/out package

datafile = pd.read_csv("C:\Users\Vivien\Desktop\\tuleysample.csv",header=None,parse_dates=[1])
og = datafile[datafile[5]=='o3']
og[6]=pd.to_numeric(og[6])
og.plot()

so2 = datafile[datafile[5]=='so2']
so2.to_csv("C:\Users\Vivien\Desktop\\tuleytest2.csv")