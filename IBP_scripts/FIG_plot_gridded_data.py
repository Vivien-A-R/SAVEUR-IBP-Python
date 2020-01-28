# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 15:43:57 2017

@author: Vivien
"""

import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#G for ground
g = pd.read_csv("C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Coordinates and Maps\DEM_Coords.txt")
gvalues = (g.grid_code*0.3048).values
gpoints = np.array([tuple(x) for x in g[['x','y']].to_records(index=False)])

xmin = gpoints[:,0].min()
xmax = gpoints[:,0].max()
ymin = gpoints[:,1].min()
ymax = gpoints[:,1].max()
grid_x, grid_y = np.mgrid[xmin:xmax:1000j,ymin:ymax:1000j]

grid_z0 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'nearest')
grid_z1 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'linear')
grid_z2 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'cubic')

plt.subplot(311)
plt.imshow(grid_z0.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
plt.title('Nearest')
plt.subplot(312)
plt.imshow(grid_z1.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
plt.title('Linear')
plt.subplot(313)
plt.imshow(grid_z2.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
plt.title('Cubic')
plt.show()

# =============================================================================
# def func(x, y):
#     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
# grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
# points = np.random.rand(1000, 2)
# values = func(points[:,0], points[:,1])
# 
# 
# grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
# 
# plt.subplot(221)
# plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
# plt.plot(points[:,0], points[:,1], 'k.', ms=1)
# plt.title('Original')
# plt.subplot(222)
# plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
# plt.title('Nearest')
# plt.subplot(223)
# plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
# plt.title('Linear')
# plt.subplot(224)
# plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
# plt.title('Cubic')
# plt.gcf().set_size_inches(6, 6)
# plt.show()
# =============================================================================
