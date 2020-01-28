# -*- coding: utf-8 -*-
"""
Created on Thu Jun 08 16:29:17 2017

@author: Vivien
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from cdf_violin import cdf #Need to update sensor_meta address in this file as well!

data_path = 'C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\'
#data_path = 'D:\Dropbox\IBP_Python\scripts_and_filenames\\'

chem_column = pd.read_table(data_path+'raw_nt_data\\soil_profiles_chemonly.csv',sep = ',')
df_soil_column = pd.read_table(data_path+'raw_nt_data\\boring_log.csv',sep = ',')

#Sensor/well metadata (for sample elevations)
sensor_meta = pd.read_table(data_path+'raw_nt_data\\wl_position_meta.csv',sep=',',index_col=False)
sensor_meta = pd.concat([sensor_meta,data_path+'processed_data\\'+sensor_meta.sensor+"_ibp_main.csv"],axis=1)
sensor_meta.columns=['data_id','top_elev_ft','cable_length_ft','ground_elev_ft','lat','long','path']

#Individual color for each soil type/color; no numeric codes
colordict = {'Black CL'     : 'black',
         'Black ML'         : 'midnightblue',
         'Brown CH'         : 'maroon',
         'Brown CL'         : 'saddlebrown',
         'Brown ML'         : 'lightcoral',
         'Brown OL'         : 'darkolivegreen',
         'Brown SP'         : 'wheat',
         'Brown Sandy CL'   : 'lightsalmon',
         'Gray CH'          : 'dimgrey',
         'Gray CL'          : 'lightsteelblue',
         'Orange SP'        : 'orangered',
         'Tan CL'           : 'tan',
         'Tan SP'           : 'goldenrod',
         }

def colorcode(x): return colordict[x]
              
def single_well(well,metal_list,log=False,l_lin = 0, l_log = 1, r_lin = 120, r_log = 1e4):
    
    f, (ax1,ax2) = plt.subplots(1,2,sharey=True,gridspec_kw = {'width_ratios':[1,4]},figsize = (8,8)) ##defining the figure

    #Plot soil type
    #These are the same every row
    ground_elev = sensor_meta[sensor_meta.data_id == well].ground_elev_ft.item()*0.3048
    xloc = 0
    xwid = 0.2
    ywid = 0.01*2.54
#    plt.ylabel("Elevation (m above MSL)") #AAAAAARGH
    
    for nn in range(0,len(df_soil_column.depth_cm)):
        #These are calculated again every row
        yloc = -1.0*df_soil_column.depth_cm[nn]/100 + ground_elev #Correct using ground elevation
        facecol=colorcode(df_soil_column[well][nn])

        ## column name = patches.Rectangle((x location, y location), rectangle width, rectangle height, edgecolor, facecolor=selects color based on the number within the data column, uses that number to select the color from the color_column variable above)
        ## the x y location represents the lower left corner of the rectangle, the y depth was assumed to in 10 cm increments but could be altered. The rectangle height will need to be based on the Y location spacing to get a representative plot.
       
        well_col = patches.Rectangle((xloc-xwid/2.,yloc),xwid,ywid,edgecolor='None',facecolor = facecol)
        ax1.add_patch(well_col)

    ax1.add_patch(patches.Rectangle((xloc-xwid/2.,yloc),xwid,1.22,edgecolor='black', fill = False))

    ax1.spines['bottom'].set_visible(False) #Remove borders
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    #ax1.yaxis.grid(True)
    #ax1.axhline(y = ground_elev,color='brown')
    ax1.set_xlim(0,0.2)
    ax1.xaxis.set_visible(False)

#==============================================================================
#     #Legend for soil type/color
#     handles = map(lambda x: colorcode(x),colordict.keys())
#     leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), colordict.keys(), handles)
#     ax1.legend(handles = leg,loc='upper left',bbox_to_anchor=(-0.4,0),ncol=5)
# 
#==============================================================================

# =============================================================================
#     #Violins
#     all_data = cdf(well)
#     
#     pos=[0.45] #xlocation to place violin plot 
#     v_fig=ax1.violinplot(all_data,pos,points=300,widths=0.2,showmeans=True,showextrema=True) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
#     for ii in v_fig['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
#         ii.set_facecolor('cornflowerblue')
#         #ii.set_edgecolor('black')
#         ii.set_alpha(1)
# #    for ii1 in v_fig['cmeans']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
# #        ii1.set_linewidth(1)
# #        ii1.set_edgecolor=('k')  
#     v_fig['cmeans'].set_linewidth(2)
#     v_fig['cmeans'].set_edgecolor('Blue')
#     v_fig['cmaxes'].set_linewidth(2)
#     v_fig['cmaxes'].set_edgecolor('Blue')
#     v_fig['cmins'].set_linewidth(2)
#     v_fig['cmins'].set_edgecolor('Blue')
#     v_fig['cbars'].set_linewidth(2)
#     v_fig['cbars'].set_edgecolor('Blue')
#     
#     ax1.set_ylim(min(ground_elev-1.24,all_data.min()-0.02),all_data.max()+0.02)
#     ax1.set_xlim(0,0.6)
#     ax1.xaxis.set_visible(False)
# =============================================================================
    

    #Plot chemical data
    #Pb_ppm	Cu_ppm	Fe_ppm	Zn_ppm	Ca_ppm   	K_ppm   	Mg_ppm  	P_ppm	        Na_ppm
    #Blue    Orange Red     Green   DYellow   Seagreen  DkOrch    FireBrick   Slateblue

    chemcols = ['royalblue','darkorange','red','green','darkgoldenrod','seagreen','darkorchid','firebrick','slateblue','black']
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    datafile = chem_column[chem_column.well_id == well]
    for analyte in metal_list:
        chem=datafile[analyte]
        col = chemcols[datafile.columns.get_loc(analyte)-2]
        if(log == True):ax2.semilogx(chem,datafile.z_m,'o-',color = col)
        else:ax2.plot(chem,datafile.z_m,'o-',color = col)

    ax2.axhline(y = ground_elev,color='brown')
    #ax2.yaxis.grid(True)
    #ax1.axhline(y = ground_elev,color='brown')
    #ax2.yaxis.grid(True)
    ax2.xaxis.tick_top()
    ax2.set_xlabel('(b): Concentration (mg/kg)') 
    ax2.xaxis.set_label_position('top')
    if(log == True):ax2.set_xlim(left = l_log,right = r_log)
    else:ax2.set_xlim(left = l_lin,right = r_lin)
    ax2.legend(labels = metal_list)

    #General plot stuff
    plt.suptitle(well)
    f.subplots_adjust(wspace=0)
    ax1.text(0.01,ground_elev + 0.05 ,"(a): Soil type\n and color")
    #ax1.text(0.3,all_data.max() + 0.05 ,"(b): Water\n level")


# =============================================================================
# for well in sensor_meta.data_id[0:10]:
#     single_well(well,["Pb_ppm","Cu_ppm","Zn_ppm"])
#     savename = "C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\figures\\"+ well + "_PbCuZn.png"
#     plt.savefig(savename)
#     plt.close()
#     single_well(well,["Fe_ppm","Mg_ppm"],log=True,l_log = 1e1)
#     savename = "C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\figures\\"+ well + "_FeMg.png"
#     plt.savefig(savename)
#     plt.close()
#     single_well(well,["K_ppm","P_ppm"],log=True)
#     savename = "C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\figures\\"+ well + "_KP.png"
#     plt.savefig(savename)
#     plt.close()
#     single_well(well,["Na_ppm"],r_lin = 200)
#     savename = "C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\figures\\"+ well + "_Na.png"
#     plt.savefig(savename)
#     plt.close()
#     single_well(well,["Ca_ppm"],log=True,l_log = 1e1)
#     savename = "C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\figures\\"+ well + "_Ca.png"
#     plt.savefig(savename)
#     plt.close()
# =============================================================================
#plt.title(title)


single_well("WLW12",["Pb_ppm","Cu_ppm","Zn_ppm"],r_lin = 180)

single_well("WLW14",["Pb_ppm","Cu_ppm","Zn_ppm"],r_lin = 180)
