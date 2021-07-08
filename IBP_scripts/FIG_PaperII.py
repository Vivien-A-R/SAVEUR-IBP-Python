# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:10:06 2017

@author: Vivien
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import seaborn as sns

import soiltype_to_numeric as sn
from cdf_violin2 import cdf

mpl.rcParams.update({'font.size': 16})

def clamp(val, minimum=0, maximum=1):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return val

def mplcolors(ctuple):
    x = [y/255. for y in ctuple]
    return x

def colorshift(ctuple,degree):
    x = [clamp(y * degree) for y in ctuple]
    return x
    

cc=[[135,206,250],[250,128,114],[173,255,47],[160,82,45],[205,175,149],[106,90,205]]
color_column = [colorshift(mplcolors(l),0.7) for l in cc]

labels = ['CL','CH','SP','ML','OL','Sandy CL']

#rpath = root_path + 'Precipitation\\'
d_path = "C:\Users\Packman-Field\Documents\Paper II\Water Data\\"

metadata = pd.read_csv(d_path + "waterlevels\\" + 'wl_position_meta.csv')
bl = sn.gcf(1)

well_ch = "WLW12"

def data_ret(resample_rate = "D"):
    #Example plotting with a chosen time series
    gw = pd.read_csv(d_path + "waterlevels\\" + well_ch + "_4yr.csv",parse_dates = ['date_time'])
    gw.loc[gw.qual_c < 1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
    gw = gw[['date_time','WS_elevation_m']]
    gw_res= gw.set_index('date_time').WS_elevation_m.resample(resample_rate).mean().to_frame().reset_index()
    datestring = "Water level processed,   dates: " + gw_res.date_time.min().strftime("%Y-%m-%d") + " - " + gw_res.date_time.max().strftime("%Y-%m-%d")
    print(datestring)
    
    sm = pd.read_csv(d_path + 'SMP2_ibp_main.csv',parse_dates = ['date_time'])
    sm_res1 = sm.set_index('date_time').a1_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_res2 = sm.set_index('date_time').a2_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_res3 = sm.set_index('date_time').a3_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_res4 = sm.set_index('date_time').a4_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_res5 = sm.set_index('date_time').a5_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_res6 = sm.set_index('date_time').a6_moisture.resample(resample_rate).mean().to_frame().reset_index()
    sm_all = pd.merge(sm_res1, sm_res2)
    sm_all = pd.merge(sm_all, sm_res3)
    sm_all = pd.merge(sm_all, sm_res4)
    sm_all = pd.merge(sm_all, sm_res5)
    sm_all = pd.merge(sm_all, sm_res6)
    sm_all.rename(columns = {'a1_moisture':'VWC%_10cm',
                         'a2_moisture':'VWC%_20cm',
                         'a3_moisture':'VWC%_40cm',
                         'a4_moisture':'VWC%_60cm',
                         'a5_moisture':'VWC%_80cm',
                         'a6_moisture':'VWC%_100cm'},inplace = True)
    datestring = "Soil moisture processed, dates: " + sm_all.date_time.min().strftime("%Y-%m-%d") + " - " + sm_all.date_time.max().strftime("%Y-%m-%d")
    print(datestring)
    
    
    rp = pd.read_csv(d_path + 'precip_in_Crete.csv', parse_dates = ['date_time'])
    rp['precip_cm'] = rp['precip_in']*2.54
    precip = rp[['date_time','precip_cm']].copy().set_index('date_time').resample(resample_rate).sum().reset_index()
    datestring = "Precipation processed,   dates: " + precip.date_time.min().strftime("%Y-%m-%d") + " - " + precip.date_time.max().strftime("%Y-%m-%d")
    print(datestring)
    
    return precip, gw_res, sm_all

a,b,c = data_ret("D") #precip, groundwater, soil moisture resampled to daily
merged = pd.merge(a, b)
merged = pd.merge(merged, c)

def stackplot():
    gelev = metadata[metadata.sensor == well_ch].ground_elev_ft.iloc[0]*0.3048
       
    bl['change'] = bl[well_ch].diff()  # colors
    fb = bl[bl['change'] != 0][['depth_cm',well_ch]]
    fb = fb.append(bl[['depth_cm',well_ch]].iloc[-1])
    fb['elevation_m'] = gelev - (fb['depth_cm'])/100
    
    f, (ax3,ax2,ax1) = plt.subplots(3, sharex=True,figsize = [14,10],gridspec_kw = {'height_ratios':[1,3,3]})
    #SM
    lw = 2
    ax1.plot(merged['date_time'],merged['VWC%_10cm'],color = color_column[0],dashes=[1,1],linewidth = lw,label = '10cm')
    ax1.plot(merged['date_time'],merged['VWC%_20cm'],color = color_column[0],dashes=[2,1],linewidth = lw,label = '20cm')
    ax1.plot(merged['date_time'],merged['VWC%_40cm'],color = color_column[0],dashes=[4,1],linewidth = lw,label = '40cm')
    ax1.plot(merged['date_time'],merged['VWC%_60cm'],color = color_column[0],dashes=[6,1],linewidth = lw,label = '60cm')
    ax1.plot(merged['date_time'],merged['VWC%_80cm'],color = color_column[1],dashes=[8,1],linewidth = lw,label = '80cm')
    ax1.plot(merged['date_time'],merged['VWC%_100cm'],color = color_column[1],dashes=[12,1],linewidth = lw,label = '100cm')
    ax1.legend(handlelength = 3,prop={'size': 12},bbox_to_anchor=(1,0.6))
    
    
    #Raw groundwater signal
    ax2.plot(merged['date_time'],merged['WS_elevation_m'],color = 'black')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    handles = map(lambda x: color_column[int(sn.soil_codes(x))-1],labels)
    leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), labels, handles)
    ax2.legend(handles = leg,prop={'size': 12},bbox_to_anchor=(1,0.6))
    
    #Raw rain signal
    ax3.plot(merged.date_time,merged.precip_cm,drawstyle = "steps",color = "black")
    ax3.set_ylim([12,0])

    #fill color
    i = 0
    while i < len(fb)-1:
        top = fb.elevation_m.iloc[i]
        bottom = fb.elevation_m.iloc[i+1]
        c = int(fb[well_ch].iloc[i]-1)
        ax2.axhspan(top,bottom,alpha = 0.3,color = color_column[c],edgecolor = None)
        i = i+1
        
    return merged


#To do: adjust to do wet/dry instead of months.    
t1_filepath = 'C:\Users\Packman-Field\Google Drive\Packman Group\Python Scripts\\IBP_scripts\\raw_nt_data\\T1_elev.csv'
def transplot():
    title = "Transect; Sand ridge to ephemeral wetland"
    xloc_dict = {       #Figure out whether I should include any others? WLW14?
            'WLW7':	0.0,
            'WLW8':	130.501,
            'WLW9':	218.29,
            'WLW4':	284.154,
            'WLW3':	433.857
            }
    #Ground elevation
    xy_elevation = pd.read_table(t1_filepath,sep=',',index_col=False)
    ground_x = (xy_elevation.x_ft*0.3048).values
    ground_y = (xy_elevation.elevation_ft*0.3048).values
    ground_err = 0.3
    #Plot boundaries
    ymin, ymax = 182.5,187.5
    xmin, xmax = -20,450
    
    fig1, ax1 = plt.subplots(figsize = (12,8)) ##defining the figure
    ax1.xaxis.set_visible(False) #Hide x-axis because it's physically meaningless
     
    for well in xloc_dict: 
        #Do columns
        #These are the same every row
        #ground_elev = metadata[metadata.sensor == well].ground_elev_ft.iloc[0]*0.3048
        xloc = xloc_dict[well]
        xwid = 0.10*(xmax-xmin)/len(xloc_dict)
        #ywid = 0.01*2.54 #one inch
        
        #Do violins
        all_data = cdf(well)
        wi = all_data[all_data.index.month.isin([1,2,3])]
        sp = all_data[all_data.index.month.isin([4,5,6])]
        su = all_data[all_data.index.month.isin([7,8,9])]
        fa = all_data[all_data.index.month.isin([10,11,12])]
        
        #Fiddled with by hand; need to find better way to determine these
        bigw = xwid*6
        wiw = bigw*0.40
        spw = bigw*0.40
        suw = bigw*0.40
        faw = bigw*0.40
        salpha = 0.6
        
        scolors = ['lightskyblue','yellowgreen','coral','gold','grey']
        slabels = ['winter','spring','summer','fall','total']
     
        pos=[xloc] #xlocation to place violin plot 
        v_fig=ax1.violinplot(all_data,pos,points=300,widths=bigw,showmeans=True,showextrema=False) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_fig['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor(scolors[4])
            ii.set_alpha(1)
            ii.set_zorder(2)
        v_fig['cmeans'].set_linewidth(2)
        v_fig['cmeans'].set_edgecolor('Black')
        
        ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
        #each has a handle so that it can be called, this way we can edit its colors and other properties
        #Winter
        v_figwi=ax1.violinplot(wi,pos,points=300,widths=wiw,showmeans=True,showextrema=False) 
        for ii in v_figwi['bodies']: 
            ii.set_facecolor(scolors[0])
            ii.set_alpha(salpha)
            ii.set_zorder(6)
        #Spring
        v_figsp=ax1.violinplot(sp,pos,points=300,widths=spw,showmeans=True,showextrema=False) 
        for ii in v_figsp['bodies']: 
            ii.set_facecolor(scolors[1])
            ii.set_alpha(salpha)
            ii.set_zorder(5)
        #Summer
        v_figsu=ax1.violinplot(su,pos,points=300,widths=suw,showmeans=True,showextrema=False) 
        for ii in v_figsu['bodies']: 
            ii.set_facecolor(scolors[2])
            ii.set_alpha(salpha)
            ii.set_zorder(4)
            
        #Fall
        v_figfa=ax1.violinplot(fa,pos,points=300,widths=faw,showmeans=True,showextrema=False) 
        for ii in v_figfa['bodies']: 
            ii.set_facecolor(scolors[3])
            ii.set_alpha(salpha)
            ii.set_zorder(3)
            
        ax1.axis([xmin,xmax,ymin,ymax]) #sets axis limits, axis does not automatically choose the best limits.
    
    plt.title(title)
    
    ax1.plot(ground_x,ground_y,color = '#7F6757')
    ax1.fill_between(ground_x, ground_y-ground_err,ground_y+ground_err,alpha = 0.2,color = '#7F6757')
    ax1.xaxis.set_visible(True)
    ax1.set_xlabel("Distance along transect (m)")
    
    wip = patches.Patch(color = scolors[0] , label = slabels[0])
    spp = patches.Patch(color = scolors[1] , label = slabels[1])
    sup = patches.Patch(color = scolors[2] , label = slabels[2])
    fap = patches.Patch(color = scolors[3] , label = slabels[3])
    top = patches.Patch(color = scolors[4] , label = slabels[4])

    handles = [wip,spp,sup,fap,top]
    labels = [h.get_label() for h in handles] 

    ax1.legend(handles=handles, labels=labels)  
    ax1.set_ylabel("Elevation (m above MSL)")
    
def sm_heatmap(p = '2'):    
    df_sm = c.copy()
    depths = [10,20,40,60,80,100]
    df_mod = df_sm.iloc[:,-6:].transpose()
    df_mod.index = depths
    df_mod.columns = df_sm['date_time'].dt.strftime('%Y-%m-%d')

    
    df_mod.loc[30] = 100
    df_mod.loc[50] = 100
    df_mod.loc[70] = 100
    df_mod.loc[90] = 100
    df_mod.sort_index(axis = 0,inplace = True)
    
    mask = df_mod.isnull()
    df_mod.replace(100,np.nan,inplace=True)
    df_mod.interpolate(method='index',inplace = True)
    
    f_sm = plt.figure()
    sns.heatmap(df_mod, mask = mask,vmin = 0, vmax = 60,cmap="YlGnBu", xticklabels = 60, cbar_kws = {'label':'Soil moisture (%VWC)'})
    f_sm.autofmt_xdate()
    plt.ylabel("Depth below ground surface (cm)")
    plt.xlabel("Date")
    plt.tight_layout()
    
    return df_mod
    
def peakstat_plot():
    df_peaks = pd.read_csv(d_path + "peakstats_thresh 0.01m.csv", index_col = 0, parse_dates = ['x_0_dt'])
    precip = a.copy()
    preceeding_rain_days = 2
    
    
    
