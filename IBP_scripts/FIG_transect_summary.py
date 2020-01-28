# -*- coding: utf-8 -*-
from cdf_violin2 import cdf
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
chem_column = pd.read_table(data_path+'raw_nt_data\\soil_profiles_chemonly.csv',sep = ',')

#Sensor/well metadata (for sample elevations)
sensor_meta = pd.read_table(data_path+'raw_nt_data\\wl_position_meta.csv',sep=',',index_col=False)
sensor_meta = pd.concat([sensor_meta,data_path+'processed_data2\\'+sensor_meta.sensor+"_ibp_main.csv"],axis=1)
sensor_meta.columns=['data_id','top_elev_ft','cable_length_ft','ground_elev_ft','lat','long','path']

df_soil_column = sn.gcf(3) #Get Coded dataFrame (from soiltype_to_numeric file)
                         
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
        ymin, ymax = 182.5,186.5
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
    fig1, ax1 = plt.subplots(figsize = (12,8)) ##defining the figure
    ax1.xaxis.set_visible(False) #Hide x-axis because it's physically meaningless
    ##For loop plots a rectangle with the fill color from the color column from above
    ## Only plots the wells included in xloc_dict; use this to do transects 
    
    #Generate legend; uses the color column and soil code dict so that if one is changed, they still agree with the legend
    labels = ['CL','CH','SP','ML','OL']
    handles = map(lambda x: color_column[int(sn.soil_codes(x))-1],labels)
    leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), labels, handles)
    plt.legend(handles = leg,loc=0)
     
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
        ax1.text(xloc-xwid,ground_elev -1.4 ,well,rotation = 60)
    
        #Do violins
        all_data = cdf(well)
    
        pos=[xloc+50] #xlocation to place violin plot 
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
        
        #Chemical data; select element in function call
        datafile = chem_column[chem_column.well_id == well]
        norm_chem_data=datafile[element]/max(datafile[element])*35+xloc+xwid/2
        ax1.add_patch(patches.Rectangle((xloc+xwid/2.,yloc),35,1.22,edgecolor='Black', facecolor = 'white'))
        ax1.plot(norm_chem_data,datafile['z_m'],'.-',color="peru") #axes
        
    ax1.axis([xmin,xmax,ymin,ymax]) #sets axis limits, axis does not automatically choose the best limits. syntax= ([xmin, xmax, ymin, ymax])
    plt.title(title)
    
    if((transect == 1) | (transect == 2)):
        ax1.plot(ground_x,ground_y,color = '#7F6757')
        ax1.fill_between(ground_x, ground_y-ground_err,ground_y+ground_err,alpha = 0.2,color = '#7F6757')
        ax1.xaxis.set_visible(True)
        ax1.set_xlabel("Distance along transect (m)")
    
    ax1.set_ylabel("Elevation (m)")



doit(1,'Cu_ppm')
doit(2,'Fe_ppm')
#doit(3)