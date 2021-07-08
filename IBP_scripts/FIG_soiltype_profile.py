# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 12:26:16 2017

@author: Vivien
"""

# -*- coding: utf-8 -*-
from cdf_violin2 import cdf
import soiltype_to_numeric as sn

import pandas as pd #pythons data/timeseries package
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###### Data i/o steps, can be changed without affecting the code, code will need to be rewritten to reflect proper structure of actual data.
data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

#Sensor/well metadata (for sample elevations)


df_soil_column = sn.gcf(1) #Get Coded dataFrame (from soiltype_to_numeric file)
                         
color_column=['lightskyblue','salmon','orchid','navajowhite','lightgreen','mediumslateblue'] ##buncha different colors that look okay together and are kind of like the real colors
#examples of the various colors <https://matplotlib.org/examples/color/named_colors.html>
#examples of patch patterns and creating rectangles <http://matthiaseisen.com/pp/patterns/p0203/> 
#more different ways to do patch patterns <https://matplotlib.org/examples/pylab_examples/hatch_demo.html>             

# locations can be given in meters. For a profile this would be the distance along the line.  Commenting out rows leaves a gap in the plot    

title = "All wells"
xloc_dict = {
    'WLW1': 0.3,
    'WLW2': 0.6,
    'WLW3': 0.9,
    'WLW4': 1.2,
    'WLW5': 1.5,
    'WLW6': 1.8,
    'WLW7': 2.1,
    'WLW8': 2.4,
    'WLW9': 2.7,
    'WLW10':3.0,
    'WLW11':3.3, 
    'WLW12':3.6,
    'WLW14':3.9
    }
#Plot boundaries
ymin, ymax = -1.5,0.5
xmin, xmax = 0,4

#Plotting     
fig1, ax1 = plt.subplots(figsize = (12,8)) ##defining the figure
ax1.xaxis.set_visible(False) #Hide x-axis because it's physically meaningless
##For loop plots a rectangle with the fill color from the color column from above
## Only plots the wells included in xloc_dict; use this to do transects 

#Generate legend; uses the color column and soil code dict so that if one is changed, they still agree with the legend
labels = ['CL','CH','SP','ML','OL','Sandy CL']
handles = map(lambda x: color_column[int(sn.soil_codes(x))-1],labels)
leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), labels, handles)
plt.legend(handles = leg,loc=0)
 
for well in xloc_dict: 
    #Do columns
    #These are the same every row
    #ground_elev = sensor_meta[sensor_meta.data_id == well].ground_elev_ft.item()*0.3048
    xloc = xloc_dict[well]
    xwid = 0.50*(xmax-xmin)/len(xloc_dict)
    ywid = 0.01*2.54 #one inch
    
    for nn in range(0,len(df_soil_column.depth_cm)):  
       #These are calculated again every row
       yloc = -df_soil_column.depth_cm[nn]/100
       facecol=color_column[df_soil_column[well][nn]-1]

        ## column name = patches.Rectangle((x location, y location), rectangle width, rectangle height, edgecolor, facecolor=selects color based on the number within the data column, uses that number to select the color from the color_column variable above)
        ## the x y location represents the lower left corner of the rectangle, the y depth was assumed to in 10 cm increments but could be altered. The rectangle height will need to be based on the Y location spacing to get a representative plot.
       
       well_col = patches.Rectangle((xloc-xwid/2.,yloc),xwid,ywid,edgecolor='None',facecolor = facecol,zorder = 1)
       ax1.add_patch(well_col)

    ax1.add_patch(patches.Rectangle((xloc-xwid/2.,yloc),xwid,1.22,edgecolor='black', fill = False))
    ax1.text(xloc-xwid, -1.3 ,well,rotation = 60)


ax1.axis([xmin,xmax,ymin,ymax]) #sets axis limits, axis does not automatically choose the best limits. syntax= ([xmin, xmax, ymin, ymax])
plt.title(title)
ax1.set_ylabel("Depth (m)")