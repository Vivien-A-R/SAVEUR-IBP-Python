# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:55:23 2017

@author: Vivien
"""
#Custom plots for Lili
# -*- coding: utf-8 -*-
from cdf_violin import cdf
import soiltype_to_numeric as sn

import pandas as pd #pythons data/timeseries package
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###### Data i/o steps, can be changed without affecting the code, code will need to be rewritten to reflect proper structure of actual data.
data_path = 'C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\'
#data_path = 'D:\IBP_Python\scripts_and_filenames\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

#Real data:
df_soil_column=pd.read_table(data_path+'raw_nt_data\\boring_log.csv',sep=',',index_col=False)
chem_column = pd.read_table(data_path+'raw_nt_data\\soil_profiles_chemonly.csv',sep = ',')
water_avg = pd.read_table(data_path +'data_products\\averages.csv',sep=',',index_col=False)

#Sensor/well metadata (for sample elevations)
sensor_meta = pd.read_table(data_path+'raw_nt_data\\wl_position_meta.csv',sep=',',index_col=False)
sensor_meta = pd.concat([sensor_meta,data_path+'processed_data\\'+sensor_meta.sensor+"_ibp_main.csv"],axis=1)
sensor_meta.columns=['data_id','top_elev_ft','cable_length_ft','ground_elev_ft','lat','long','path']
sensor_meta = sensor_meta.loc[0:10,:]


df_soil_column,b,c = sn.gcf(1) #Get Coded dataFrame (from soiltype_to_numeric file)
                         
color_column=['lightskyblue','salmon','orchid','navajowhite','lightgreen'] ##buncha different colors that look okay together and are kind of like the real colors
#examples of the various colors <https://matplotlib.org/examples/color/named_colors.html>
#examples of patch patterns and creating rectangles <http://matthiaseisen.com/pp/patterns/p0203/> 
#more different ways to do patch patterns <https://matplotlib.org/examples/pylab_examples/hatch_demo.html>             

# locations can be given in meters. For a profile this would be the distance along the line.  Commenting out rows leaves a gap in the plot    

#Transect 1, Sand ridge, no ditches; Transect 2, Pond and upland
def doit(transect,element):
    if(transect==1):
        title = "Transect 1; Sand ridge to pond"
        xloc_dict = {
                'WLW7':	0.0,
                'WLW8':	130.501,
                'WLW9':	218.29,
                'WLW4':	284.154,
                #'WLS2':	348.684, #Use this elsewhere
                'WLW3':	433.857
                }
        #Ground elevation
        xy_elevation = pd.read_table(data_path+'raw_nt_data\\T1_elev.csv',sep=',',index_col=False)
        ground_x = (xy_elevation.x_ft*0.3048).values
        ground_y = (xy_elevation.elevation_ft*0.3048).values
        ground_err = 0.25
        #Plot boundaries
        ymin, ymax = 182.5, 187.5
        xmin, xmax = -20, 500
    elif(transect == 2):
        title = "Transect 2; Pond and upland"
        xloc_dict = {
                'WLW1':	0,
                'WLW2':	84.694,
                #'WLS2':	185.479,
                'WLW5':	282.956,
                #'WLS4':	335.561,
                'WLW6':	398.091,
                'WLW10':	688.601,
                }
        #Ground elevation
        xy_elevation = pd.read_table(data_path+'raw_nt_data\\T2_elev.csv',sep=',',index_col=False)
        ground_x = (xy_elevation.x_ft*0.3048).values
        ground_y = (xy_elevation.elevation_ft*0.3048).values
        ground_err = 0.3
        #Plot boundaries
        ymin, ymax = 182.5,187
        xmin, xmax = -20,750
    else:
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
            'WLW11':3.3 
            }
        #Plot boundaries
        ymin, ymax = 182.5,187.5
        xmin, xmax = 0,4
    
    #Plotting     
    #fig1, ax1 = plt.subplots(figsize = (14,8)) ##defining the figure
    fig1, (ax1,ax2) = plt.subplots(1,2,figsize = (14,8),gridspec_kw = {'width_ratios':[5,2]},sharey=True) ##defining the figure
    #fig1, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(1,6,figsize = (16,8),sharey=True,gridspec_kw = {'width_ratios':[5,1,1,1,1,1]}) ##defining the figure
    #ax1.xaxis.set_visible(False) #Hide x-axis because it's physically meaningless
    ##For loop plots a rectangle with the fill color from the color column from above
    ## Only plots the wells included in xloc_dict; use this to do transects 
    
#==============================================================================
#     #Generate legend; uses the color column and soil code dict so that if one is changed, they still agree with the legend
#     labels = ['CL','CH','SP','ML','OL']
#     handles = map(lambda x: color_column[int(soil_codes(x))-1],labels)
#     leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), labels, handles)
#     ax1.legend(handles = leg,loc = 'upper left')
#==============================================================================
     
    for well in xloc_dict: 
        #Do columns
        #These are the same every row
        ground_elev = sensor_meta[sensor_meta.data_id == well].ground_elev_ft.item()*0.3048
        xloc = xloc_dict[well]
        xwid = 0.10*(xmax-xmin)/len(xloc_dict)
        ywid = 0.01*2.54 #one inch
        
        for nn in range(0,len(df_soil_column.depth_cm)):  
           #These are calculated again every row
           yloc = -1.0*df_soil_column.depth_cm[nn]/100 + ground_elev #Correct using ground elevation
           facecol=color_column[df_soil_column[well][nn]-1]
    
            ## column name = patches.Rectangle((x location, y location), rectangle width, rectangle height, edgecolor, facecolor=selects color based on the number within the data column, uses that number to select the color from the color_column variable above)
            ## the x y location represents the lower left corner of the rectangle, the y depth was assumed to in 10 cm increments but could be altered. The rectangle height will need to be based on the Y location spacing to get a representative plot.
           
           well_col = patches.Rectangle((xloc-xwid/2.,yloc),xwid,ywid,edgecolor='None',facecolor = facecol,zorder = 1)
           ax1.add_patch(well_col)
    
        ax1.add_patch(patches.Rectangle((xloc-xwid/2.,yloc),xwid,1.22,edgecolor='black', fill = False))
        #ax1.text(xloc-xwid,ground_elev -1.4 ,well,rotation = 60)
    
        #Do violins
        all_data = cdf(well)
    
        pos=[xloc+20] #xlocation to place violin plot 
        if((transect == 1) | (transect == 2)):
            v_fig=ax1.violinplot(all_data,pos,points=300,widths=xwid*1.5,showmeans=True,showextrema=True) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_fig['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor('cornflowerblue')
            #ii.set_edgecolor('black')
            ii.set_alpha(0.8)
            ii.set_zorder(2)
    #    for ii1 in v_fig['cmeans']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
    #        ii1.set_linewidth(1)
    #        ii1.set_edgecolor=('k') 
        v_fig['cmeans'].set_linewidth(2)
        v_fig['cmeans'].set_edgecolor('Blue')
        v_fig['cmaxes'].set_linewidth(2)
        v_fig['cmaxes'].set_edgecolor('Blue')
        v_fig['cmins'].set_linewidth(2)
        v_fig['cmins'].set_edgecolor('Blue')
        v_fig['cbars'].set_linewidth(2)
        v_fig['cbars'].set_edgecolor('Blue')
        ax1.text(xloc-xwid,min(min(all_data)-0.2,ground_elev -1.4) ,well,rotation = 60)
#==============================================================================
#         
#         #Chemical data
#         datafile = cudata[cudata.well == well]
#         norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#         chem_el = ground_elev-datafile.cum_depth_cm/100
#         ax1.add_patch(patches.Rectangle((xloc+xwid/2.,yloc),35,1.22,edgecolor='Black', facecolor = 'white'))
#         ax1.plot(norm_chem_data,chem_el,'.-',color="peru",label = "Copper conc.") #axes
#         
#==============================================================================
        #Chem data on side plot
        datafile = chem_column[chem_column.well_id == well]
        #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
        ax2.plot(datafile[element],datafile.z_m,'.-',label = well) #axes
        ax2.set_xlabel("Copper Conc. (mg/kg)")
        ax2.xaxis.set_label_position('top')
        ax2.xaxis.tick_top()
        ax2.set_xlim(xmin = 0,xmax = 50)
    
        
    ax1.axis([xmin,xmax,ymin,ymax]) #sets axis limits, axis does not automatically choose the best limits. syntax= ([xmin, xmax, ymin, ymax])
    
    #plt.title(title)
    
    if((transect == 1) | (transect == 2)):
        ax1.plot(ground_x,ground_y,color = '#7F6757')
        ax1.fill_between(ground_x, ground_y-ground_err,ground_y+ground_err,alpha = 0.2,color = '#7F6757')
        ax1.xaxis.set_visible(True)
        ax1.set_xlabel("Distance along transect (m)")
    
    ax1.set_ylabel("Elevation (m)")
    plt.subplots_adjust(wspace=0.0001)

#==============================================================================
#     #WLW1,2,5,6,10
#     #Chem data on multiple side plots
#     w = "WLW1"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax2.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax2.xaxis.tick_top()
#     ax2.set_xticks((0,10,20))
#     ax2.set_xlim(xmin = 0,xmax = 25)
#     ax2.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax2.text(1,186.3,w)
#     
#     w = "WLW2"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax3.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax3.xaxis.tick_top()
#     ax3.set_xticks((0,10,20))
#     ax3.set_xlim(xmin = 0,xmax = 25)
#     ax3.set_xlabel("Copper Conc. (mg/kg)")
#     ax3.xaxis.set_label_position('top')
#     ax3.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax3.text(1,186.3,w)
#     
#     w = "WLW5"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax4.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax4.xaxis.tick_top()
#     ax4.set_xticks((0,10,20))
#     ax4.set_xlim(xmin = 0,xmax = 25)
#     ax4.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax4.text(1,186.3,w)
#     
#     w = "WLW6"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax5.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax5.xaxis.tick_top()
#     ax5.set_xticks((0,10,20))
#     ax5.set_xlim(xmin = 0,xmax = 25)
#     ax5.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax5.text(1,186.3,w)
#     
#     w = "WLW10"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax6.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax6.xaxis.tick_top()
#     ax6.set_xticks((0,10,20))
#     ax6.set_xlim(xmin = 0,xmax = 25)
#     ax6.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax6.text(1,186.3,w)
#==============================================================================
    
#==============================================================================
#     #WLW7,8,9,4,3
#     #Chem data on multiple side plots
#     w = "WLW7"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax2.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax2.xaxis.tick_top()
#     ax2.set_xticks((0,20,40))
#     ax2.set_xlim(xmin = 0,xmax = 50)
#     ax2.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax2.text(1,187,w)
# 
#     
#     w = "WLW8"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax3.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax3.xaxis.tick_top()
#     ax3.set_xticks((0,20,40))
#     ax3.set_xlim(xmin = 0,xmax = 50)
#     ax3.set_xlabel("Copper Conc. (mg/kg)")
#     ax3.xaxis.set_label_position('top')
#     ax3.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax3.text(1,187,w)
# 
#     
#     w = "WLW9"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax4.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax4.xaxis.tick_top()
#     ax4.set_xticks((0,20,40))
#     ax4.set_xlim(xmin = 0,xmax = 50)
#     ax4.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax4.text(1,187,w)
# 
#     
#     w = "WLW4"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax5.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax5.xaxis.tick_top()
#     ax5.set_xticks((0,20,40))
#     ax5.set_xlim(xmin = 0,xmax = 50)
#     ax5.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax5.text(1,187,w)
# 
#     
#     w = "WLW3"
#     datafile = cudata[cudata.well == w]
#     ground_elev = sensor_meta[sensor_meta.data_id == w].ground_elev_ft.item()*0.3048
#     #norm_chem_data=datafile.cu_ppm/max(datafile.cu_ppm)*35+xloc+xwid/2
#     chem_el = ground_elev-datafile.cum_depth_cm/100
#     ax6.plot(datafile.cu_ppm,chem_el,'.-',label = w,color = 'saddlebrown') #axes
#     ax6.xaxis.tick_top()
#     ax6.set_xticks((0,20,40))
#     ax6.set_xlim(xmin = 0,xmax = 50)
#     ax6.axhline(y=water_avg[water_avg.data_id == w].average_summer.item())
#     ax6.text(1,187,w)
# 
#     
#==============================================================================



doit(1,'Cu_ppm')