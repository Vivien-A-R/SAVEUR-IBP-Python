# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 11:13:39 2017

@author: Vivien
"""
import pandas as pd
import seaborn as sns

#Setup grid
swx = -87.69464583
nex = -87.68221778
swy = 41.60442222
ney = 41.60827778
xm = 1034
ym = 428
mdx = xm/(nex-swx)
mdy = ym/(ney-swy)

#Get file
p = pd.read_csv('D:\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Coordinates and Maps\DEM_coords.txt')
p['elevation_m'] = p['grid_code']*0.3048
p.drop(['grid_code','FID','pointid'],axis = 1,inplace = True)

#Process file
p['xg'] = (p['x']-swx)*mdx
p['yg'] = (p['y']-swy)*mdy
p = p[['xg','yg','elevation_m']]
p2 = p.pivot_table("elevation_m",'xg','yg').dropna(axis = 1)
p2 = p2.drop(p2.columns[0], axis=1)
sns.heatmap(p2,square = True,xticklabels = 10, yticklabels = 10) #Test plot
p.to_csv("C:\Users\Vivien\Desktop\TABULAR_DEM.txt",index = None,sep = '\t') #Save for import
p2.to_csv("C:\Users\Vivien\Desktop\GRID_DEM.txt",index = None,sep = '\t') #Save for import