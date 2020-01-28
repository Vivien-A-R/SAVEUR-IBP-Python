# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:44:02 2016
new mostly commented script that doesn't include all of the coding (much of it unrelated) of the larger script that this was a part of originally
@author: colinbphillips
"""

import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
#import os as os #d
import matplotlib.pyplot as plt
#import pylab as pylab
import scipy.stats as scs
import scipy as sc
from scipy.optimize import curve_fit
import matplotlib as mpl

mpl.rcParams.update({'font.size': 18})
pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20
#plt.style.use('cbp_presentation')
###############################################################################
#data_path = 'C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\'
data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'
sensor_meta = pd.read_table(data_path+'wl_position_meta.csv',sep=',',index_col=False)
sensor_meta = pd.concat([sensor_meta,data_path+sensor_meta.sensor+"_ibp_main.csv"],axis=1)
sensor_meta.columns=['data_id','top_elev_ft','cable_length_ft','ground_elev_ft','lat','long','date','path']

###################################################################################
def cdf(data_id,plotcdf=False,plotall=False):
    groundlevel = sensor_meta[sensor_meta.data_id == data_id].ground_elev_ft.item()*0.3048
    #sensor_elev = sensor_meta[sensor_meta.data_id == data_id].top_elev_ft.item()*0.3048 - sensor_meta[sensor_meta.data_id == data_id].cable_length_ft.item()*0.3048
    file_path = sensor_meta[sensor_meta.data_id == data_id].path.item()
    
    df_wl_o=pd.read_table(file_path,sep=',',index_col=0,parse_dates=['date_time']) #loads data for the well
    df_wl=df_wl_o[df_wl_o.WS_elevation_m < groundlevel] #selects data below ground level only
    
    ############################## all of the fitting commands take place below here.
    cdf_data=pd.DataFrame() ##initialize a dataframe for storing cumulative distribution function (cdf)
    cdf_data['depth']=df_wl.depth_m.sort_values(axis=0,ascending=True)
    cdf_data['cdf']=np.arange(1.0,len(df_wl)+1,1)/len(df_wl) ##creates CDF including zero data
    well_bottom = 0.01 #Why this value?
    cdf_data=cdf_data[cdf_data.depth > well_bottom]##remove noise data close to bottom of the well. can be expanded if noise is larger then 1 cm in depth
    data_x=cdf_data.depth #xdata to fit, something of an uncessary step given that the data is unchanged from the cdf_data, but can be useful if you want to tweak the data here for other functional fits.
    data_y=cdf_data.cdf #ydata to fit
    #print max(cdf_data.depth+sensor_elev)
    
    #fitting normal distribution
    guess_mean=data_x.mean() #we need to provide the curve_fit function with a guess of the mean and std, well use the mean and std from the depth data that is above the base of the well.
    guess_sigma=data_x.std() #guess at std
    def line(data_x,mu,sigma): ##data_x is data_x from above, mu is a placeholder for the mean, sigma is a placeholder for the standard deviation. Defining the function these are just dummy variables.
        return 0.5*(1+sc.special.erf((data_x-mu)/(sigma*2**.5)))    ### the function of a normal cdf. erf = error function
    
    popt, pcov = curve_fit(line, data_x, data_y, p0=(guess_mean,guess_sigma))
    ## note. popt = [mean, std] estimated from the functional fit.
    ## note. pcov is the covariance matrix, used to assess goodness of fit, which we aren't doing here.
    
    #####If we are only after the mean water level (below ground) then the first term in popt is all we need. popt[0] ~ mean(below ground water level) for wells with complete or nearly complete records
    #####it batch processing you can create a new dataframe and store the popt data and then save that as a text file and have all of the mean water levels and their standard deviation.
    x=np.linspace(df_wl.depth_m.min()-1,df_wl.depth_m.max(),200) #x line for plotting, can be extended for both min and max if needed.
    p=scs.norm.cdf(x,popt[0],popt[1]) #computes normal probabilities given the fitted mean and std from the curve_fit. used for plotting.
    
    if(plotcdf == True):
        plt.subplots() #initializes figure
        #plt.plot(cdf_data.depth+sensor_elev,cdf_data.cdf,'+',label = 'cdf')
        #plt.plot(x+sensor_elev,p,'r--',linewidth=3,label = 'normal distribution fit') #plots the normal cdf.
        #plt.xlim(xmin = sensor_elev-0.5,xmax = groundlevel+0.2)
        
        plt.plot(cdf_data.depth,cdf_data.cdf,'+',label = 'cdf')
        plt.plot(x,p,'r--',linewidth=3,label = 'normal distribution fit') #plots the normal cdf.
        plt.xlim(xmin = -0.5,xmax = 1.2)
        plt.ylabel('cdf')
        plt.xlabel('Elevation (m)')
        plt.title(data_id)
        #plt.text(sensor_elev,0.1,"*") #Mark sensor elevation for later use.
        plt.legend(loc = 2)
            
    #plt.figure()
    #plt.plot(df_wl_o.date_time,df_wl_o.depth_m,'-') #plots time and depth just to inspect.
    #plt.ylabel('depth (m)')
    #plt.xlabel('time')
    
    lower_depth=scs.norm.ppf(0.02,loc=popt[0],scale=popt[1]) ## sets the lower depth as the depth where the CDF has a probability of 0.02 (this is akin to truncating the distribution for very low values of the CDF)
    nbins=np.round(np.sqrt(len(df_wl))) ##number of bins from for the below ground dataset if we were to create a histogram
    bin_spacing=np.abs(df_wl.depth_m.max()-lower_depth)/nbins #bin spacing
    bin_midpoints=np.arange(lower_depth+(bin_spacing/2.0),df_wl.depth_m.max(),(bin_spacing),dtype='float') ##the midpoints of the bins
    bin_lower=bin_midpoints[bin_midpoints <= well_bottom] ##the bin midpoints below the bottom of the well
    
    if(len(bin_lower) != 0):
        p_prediction=scs.norm.pdf(bin_lower,popt[0],popt[1]) ##normalized probability density for the bin midpoints
        n_samples=(np.round(p_prediction*len(df_wl)*bin_spacing)).astype('int') #number of samples per bin.
        s_add=pd.Series(np.ones([n_samples[0]])*bin_midpoints[0],index=np.arange(n_samples[0])) #initial series for data in first bin
        
        for ni in range(1,len(bin_lower)): ##for loop repeats and the process used to create the initial missing data in s_add above.  
            temp_data=pd.Series(np.ones([n_samples[ni]])*bin_midpoints[ni],index=np.arange(n_samples[ni]))
            s_add=s_add.append(temp_data, ignore_index = True)
            
        bottom_el = (sensor_meta[sensor_meta.data_id == data_id].top_elev_ft.iloc[0]-sensor_meta[sensor_meta.data_id == data_id].cable_length_ft.iloc[0])*.3048
        z_add = s_add + bottom_el

        data_depth=df_wl_o[df_wl_o.depth_m > well_bottom].WS_elevation_m #extracting all of the elevation data from the original well data greater then the bottom of the well        
        all_data = z_add.append(data_depth, ignore_index = True) ##combines the calculated missing data with the actual known data. The missing data (data below the well sensor) is only 'correct' in a statistical sense and is not representative of the time series.

        #Might want to randomly intersperse the fill data into the gaps?
        gaptimes = df_wl_o.loc[df_wl_o.depth_m < well_bottom].index
        wl_na = df_wl_o.loc[:,'WS_elevation_m'].copy()
        wl_na.loc[gaptimes] = np.nan
        z_shuff = z_add.sample(frac = 1)
        if len(z_shuff) < len(gaptimes):
            z_shuff = z_shuff.append(pd.Series(np.random.choice(z_shuff,size = len(gaptimes)-len(z_shuff),replace = True)))
        if len(z_shuff) > len(gaptimes):
            z_shuff = z_shuff[0:len(gaptimes)]
        z_shuff.index = gaptimes
        wl_na.update(z_shuff)
        all_data = wl_na.copy() ## 
    else: all_data=df_wl_o[df_wl_o.depth_m > well_bottom].WS_elevation_m
    if(plotall == True): 
        plt.figure()
        plt.plot(all_data)
    return all_data

#e,f = othershit(a,b,c,d)

data_id = "WLW9"
groundlevel = sensor_meta[sensor_meta.data_id == data_id].ground_elev_ft.item()*0.3048
#sensor_elev = sensor_meta[sensor_meta.data_id == data_id].top_elev_ft.item()*0.3048 - sensor_meta[sensor_meta.data_id == data_id].cable_length_ft.item()*0.3048
file_path = sensor_meta[sensor_meta.data_id == data_id].path.item()

df_wl_o=pd.read_table(file_path,sep=',',index_col=0,parse_dates=['date_time']) #loads data for the well
test = df_wl_o.loc[:,'WS_elevation_m'].copy()

def vioplot(all_data):
    fig1, ax1 = plt.subplots(1,1) #more advanced plotting routine
    pos=[1] #xlocation to place violin plot 
    v_fig=ax1.violinplot(all_data,pos,points=300,widths=0.5,showmeans=True,showextrema=True) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
    for ii in v_fig['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
        ii.set_facecolor('cornflowerblue')
        #ii.set_edgecolor('black')
        ii.set_alpha(1)
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
    ax1.set_xlim([.25,1.75])
    ax1.set_ylim([182,187])
    
# =============================================================================
# wells = ['WLW1','WLW2','WLW3','WLW4','WLW5','WLW6','WLW7','WLW8','WLW9','WLW10']
# for well in wells:
#     x = cdf(well,True)
#     vioplot(x)
#     print(well)
# =============================================================================
    
#==============================================================================
#     ##placing an inset or plot within other plots, though we may not need to deal with this at all, we can just place the violin plots on the main figure using the pos as the x_location of the well.
#     left, bottom, width, height =[0.2, 0.6, 0.2, 0.25] #left edge of the plot as a fraction of the current plot space (.2 is then 20% of the plot width from the left edge of the main plot), bottom is the same idea, height and width are also in percentages relative to the primary plot
#     ax1_in = fig1.add_axes([left, bottom, width, height]) ##adds the inset
#     s_fig=ax1_in.violinplot(all_data,pos,points=300,widths=0.5,showmeans=True,showextrema=True) #an addtional violin plot for the example
#     #ax1_in.set_axis([])
#     ax1_in.set_xlim([.25,1.75]) #changing the axis
#     ax1_in.set_ylim([182,187])
#     ax1_in.axis('off') #turning off the inset's axis.
#==============================================================================