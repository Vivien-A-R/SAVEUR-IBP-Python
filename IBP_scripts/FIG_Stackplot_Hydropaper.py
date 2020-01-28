# -*- coding: utf-8 -*-
"""
Pretty pretty plots for GMP Hydro paper
Cannibalized from AGU 2017 plot scripts and precip plot scripts
Created on Thu Nov 30 15:10:06 2017

@author: Vivien
"""

import pandas as pd
import numpy as np

#import seaborn.apionly as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

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

root_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\\'
gwpath = root_path + 'Processed Water Level Data\\'
smpath = root_path + 'Deprecated Data Folder\\Data From SMP\\'
rpath = root_path + 'Deprecated Data Folder\\Precipitation\\'

metadata = pd.read_csv(gwpath + 'wl_position_meta.csv')
bl = sn.gcf(1)


def stackplot():
    #Example plotting with a chosen time series
    well_ch = "WLW3"
    gw = pd.read_csv(gwpath + well_ch + "_ibp_main.csv",parse_dates = ['date_time'])
    gw.loc[gw.qual_c < 1,"WS_elevation_m"] = np.nan #Skip qc-flagged values
    gw = gw[['date_time','WS_elevation_m']]
    gw_res= gw.set_index('date_time').WS_elevation_m.resample("D").mean().to_frame().reset_index()
    gw_res.rename(columns = {'WS_elevation_m':'gw_elev_m'},inplace = True)
    
    sens_ch = "WLS1"
    sw = pd.read_csv(gwpath + sens_ch + "_ibp_main.csv",parse_dates = ['date_time'])
    sw.loc[sw.qual_c < 1, "WS_elevation_m"] = np.nan
    sw = sw[['date_time','WS_elevation_m']]
    sw_res= sw.set_index('date_time').WS_elevation_m.resample("D").mean().to_frame().reset_index()
    sw_res.rename(columns = {'WS_elevation_m':'sw_elev_m'},inplace = True)
    
    sm = pd.read_csv(smpath + 'SMP2_ibp_main.csv',parse_dates = ['date_time'])
    sm_res1 = sm.set_index('date_time').a1_moisture.resample("D").mean().to_frame().reset_index()
    sm_res2 = sm.set_index('date_time').a2_moisture.resample("D").mean().to_frame().reset_index()
    sm_res3 = sm.set_index('date_time').a3_moisture.resample("D").mean().to_frame().reset_index()
    sm_res4 = sm.set_index('date_time').a4_moisture.resample("D").mean().to_frame().reset_index()
    sm_res5 = sm.set_index('date_time').a5_moisture.resample("D").mean().to_frame().reset_index()
    sm_res6 = sm.set_index('date_time').a6_moisture.resample("D").mean().to_frame().reset_index()
    
    
    rp = pd.read_csv(rpath + 'PRECIP_TEMP_IBP_filled_15min.csv',parse_dates = ['date_time'],index_col = 0)
    rp['precip_cm'] = rp['incr_in']*2.54
    rp_IBP = rp.copy()
    rp_IBP.loc[rp_IBP['qc'] == -1,"precip_cm"] = np.nan
    rp_CW = rp.copy()
    rp_CW.loc[rp_CW['qc'] == 1,"precip_cm"] = np.nan
    
    merged = pd.merge(rp_IBP[['date_time','precip_cm']],rp_CW[['date_time','precip_cm']],on = 'date_time',suffixes = ['_IBP','_CW'])
    merged = pd.merge(merged, gw_res)
    merged = pd.merge(merged, sw_res)
    merged = pd.merge(merged, sm_res1)
    merged = pd.merge(merged, sm_res2)
    merged = pd.merge(merged, sm_res3)
    merged = pd.merge(merged, sm_res4)
    merged = pd.merge(merged, sm_res5)
    merged = pd.merge(merged, sm_res6)
    
    merged.rename(columns = {'a1_moisture':'VWC%_10cm',
                             'a2_moisture':'VWC%_20cm',
                             'a3_moisture':'VWC%_40cm',
                             'a4_moisture':'VWC%_60cm',
                             'a5_moisture':'VWC%_80cm',
                             'a6_moisture':'VWC%_100cm'},inplace = True)
    
    gelev = metadata[metadata.sensor == well_ch].ground_elev_ft.iloc[0]*0.3048
    
    bl['change'] = bl[well_ch].diff()
    fb = bl[bl['change'] != 0][['depth_cm',well_ch]]
    fb = fb.append(bl[['depth_cm',well_ch]].iloc[-1])
    fb['elevation_m'] = gelev - (fb['depth_cm'])/100
    
    f, (ax3,ax4,ax2,ax1) = plt.subplots(4, sharex=True,figsize = [14,10],gridspec_kw = {'height_ratios':[1,1,1,1]})

    #soil moisture
    lw = 2
    ax1.plot(merged['date_time'],merged['VWC%_10cm'],color = color_column[0],dashes=[1,1],linewidth = lw,label = '10cm')
    ax1.plot(merged['date_time'],merged['VWC%_20cm'],color = color_column[0],dashes=[2,1],linewidth = lw,label = '20cm')
    ax1.plot(merged['date_time'],merged['VWC%_40cm'],color = color_column[0],dashes=[4,1],linewidth = lw,label = '40cm')
    ax1.plot(merged['date_time'],merged['VWC%_60cm'],color = color_column[0],dashes=[6,1],linewidth = lw,label = '60cm')
    ax1.plot(merged['date_time'],merged['VWC%_80cm'],color = color_column[1],dashes=[8,1],linewidth = lw,label = '80cm')
    ax1.plot(merged['date_time'],merged['VWC%_100cm'],color = color_column[1],dashes=[12,1],linewidth = lw,label = '100cm')
    ax1.legend(handlelength = 3,prop={'size': 12},bbox_to_anchor=(1,1.05))

    #Raw groundwater signal
    ax2.plot(merged['date_time'],merged['gw_elev_m'],color = 'black')
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    handles = map(lambda x: color_column[int(sn.soil_codes(x))-1],labels)
    leg = map(lambda labels, handles: patches.Patch(color = handles, label = labels), labels, handles)
    ax2.legend(handles = leg,prop={'size': 12},bbox_to_anchor=(1,1.05))
    
    #Raw rain signal
    ax3.step(merged.date_time,merged.precip_cm_IBP,color = 'black')
    ax3.step(merged.date_time,merged.precip_cm_CW,color = 'darkgray')
    ax3.set_ylim([0.4,0])
    
    #Raw surface signal
    ax4.plot(merged.date_time,merged.sw_elev_m)
    ax4.set_ylim([184.0,185.0])

    #fill color
    i = 0
    while i < len(fb)-1:
        top = fb.elevation_m.iloc[i]
        bottom = fb.elevation_m.iloc[i+1]
        c = int(fb[well_ch].iloc[i]-1)
        ax2.axhspan(top,bottom,alpha = 0.3,color = color_column[c],edgecolor = None)
        i = i+1
    return merged
        
def slope(): 
    mash = pd.read_csv("mashed_test.csv",parse_dates = ['date_time'])
    xloc_dict = {
            'WLW7':	0.0,
            'WLW8':	130.501,
            'WLW9':	218.29,
            'WLW4':	284.154,
            'WLS2':	348.684,
            'WLW3':	433.857
            }
    
    colors = ['red','orange','yellow','yellowgreen','green','lime','cyan','aqua','blue','violet','purple','magenta','pink']
    colors = ['blue','yellowgreen','red','orange']
    
    fig = plt.figure()
    t = [item for item in xloc_dict.keys()]
    tnames = [s + '_WS_elevation_m' for s in t]
    df_t = mash[['date_time']+tnames]
    qcnames = [s + '_qual_c' for s in t]
    df_t = mash[['date_time']+tnames+qcnames]
    
    for sensor in t:
        df_t.loc[df_t[sensor+'_qual_c'] < 1, sensor + "_WS_elevation_m"] = np.nan
        
    df_t = df_t.drop(qcnames,axis = 1)
    df_t.columns = ['date_time'] + t
    
    df_t_res = df_t.resample("4H",on='date_time').mean().reset_index()
    numc = len(df_t_res)
    slopes = []
    vars = []
    c = numc
    for index, row in df_t_res.iterrows():
        row = row.to_frame()
        month = pd.to_datetime(row.loc['date_time'].values[0]).month
        row = row.drop(['date_time'],axis = 0).reset_index()
        row.columns = ['sensor','elev']
        row['pos'] = row['sensor'].map(xloc_dict)
        row = row.sort_values(by = 'pos')
        c = c - 1.
        f_lr = row.dropna()
        n = len(f_lr)
        if n < 3: 
            b = np.nan
            var = np.nan
        else:
            sxy = sum(f_lr.pos*f_lr.elev)
            sx = sum(f_lr.pos)
            sy = sum(f_lr.elev)
            sxsq = sum(f_lr.pos**2)
            sysq = sum(f_lr.elev**2)
            a = (sy*sxsq-sx*sxy)/(n*sxsq-sx**2)
            b = (n*sxy - sx*sy)/(n*sxsq-sx**2)
            meany = f_lr.elev.mean()
            var = sum((f_lr.elev - meany)**2)/n
            
        plt.plot(row['pos'],a+row['pos']*b, color = colors[month%4])
        slopes = slopes + [b]
        vars = vars + [var]
        
    slopes = np.asarray(slopes)
    vars = np.asarray(vars)
    df_t_res['slopes']  = -slopes
    df_t_res['vars']  = vars
    df_t_res.plot(x = 'date_time',y = ['slopes'])
    plt.show()
    df_t_crop = df_t_res.loc[df_t_res['date_time'] > '20170115']
    df_t_crop = df_t_crop.loc[df_t_res['date_time'] < '20170601']    
    df_t_crop.plot(x = 'date_time',y = ['slopes'])
    
t1_filepath = 'C:\Users\Packman-Field\Google Drive\Packman Group\Python Scripts\\IBP_scripts\\raw_nt_data\\T1_elev.csv'
def transplot():
    title = "Transect; Sand ridge to wetland"
    xloc_dict = {
            'WLW7':	0.0,
            'WLW8':	130.501,
            'WLW9':	218.29,
            'WLW4':	284.154,
            #'WLS2':	348.684, #Use this elsewhere
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
    ##For loop plots a rectangle with the fill color from the color column from above
    ## Only plots the wells included in xloc_dict; use this to do transects 
     
    for well in xloc_dict: 
        #Do columns
        #These are the same every row
        ground_elev = metadata[metadata.sensor == well].ground_elev_ft.iloc[0]*0.3048
        xloc = xloc_dict[well]
        xwid = 0.10*(xmax-xmin)/len(xloc_dict)
        ywid = 0.01*2.54 #one inch
        
# =============================================================================
#         for nn in range(0,len(bl.depth_cm)):  
#            #These are calculated again every row
#            yloc = -1.0*bl.depth_cm[nn]/100 + ground_elev #Correct using ground elevation
#            facecol=color_column[bl[well][nn]-1]
#     
#             ## column name = patches.Rectangle((x location, y location), rectangle width, rectangle height, edgecolor, facecolor=selects color based on the number within the data column, uses that number to select the color from the color_column variable above)
#             ## the x y location represents the lower left corner of the rectangle, the y depth was assumed to in 10 cm increments but could be altered. The rectangle height will need to be based on the Y location spacing to get a representative plot.
#            
#            well_col = patches.Rectangle((xloc-xwid/2.,yloc),xwid,ywid,edgecolor='None',facecolor = facecol,zorder = 1)
#            ax1.add_patch(well_col)
#     
#         ax1.add_patch(patches.Rectangle((xloc-xwid/2.,yloc),xwid,1.22,edgecolor='black', fill = False))
#         ax1.text(xloc-xwid,ground_elev -1.4 ,well,rotation = 60)
# =============================================================================
    
        #Do violins
        all_data = cdf(well)
        gw = pd.read_csv(gwpath + well + "_ibp_main.csv",parse_dates = ['date_time'],index_col = 0)
        gw.loc[gw.qual_c < 1,"WS_elevation_m"] = np.nan #Skip qc-flagged values
        all_data = gw.loc[:,'WS_elevation_m'].dropna().copy()
        
        wi = all_data[all_data.index.month.isin([1,2,3])]
        sp = all_data[all_data.index.month.isin([4,5,6])]
        su = all_data[all_data.index.month.isin([7,8,9])]
        fa = all_data[all_data.index.month.isin([10,11,12])]
        
        bigw = xwid*6
        wiw = bigw*0.40
        spw = bigw*0.32
        suw = bigw*0.22
        faw = bigw*0.35
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
        
        v_figwi=ax1.violinplot(wi,pos,points=300,widths=wiw,showmeans=True,showextrema=False) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_figwi['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor(scolors[0])
            ii.set_alpha(salpha)
            ii.set_zorder(6)
        v_figsp=ax1.violinplot(sp,pos,points=300,widths=spw,showmeans=True,showextrema=False) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_figsp['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor(scolors[1])
            ii.set_alpha(salpha)
            ii.set_zorder(5)
        v_figsu=ax1.violinplot(su,pos,points=300,widths=suw,showmeans=True,showextrema=False) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_figsu['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor(scolors[2])
            ii.set_alpha(salpha)
            ii.set_zorder(4)
        v_figfa=ax1.violinplot(fa,pos,points=300,widths=faw,showmeans=True,showextrema=False) #giving the violin plot a handle so that it can be called, this way we can edit its colors and other properties
        for ii in v_figfa['bodies']: ##violin plots have a lot going on so the routine to change things requires a for loop. <https://matplotlib.org/devdocs/gallery/statistics/customized_violin.html> for instructions and examples follow that web link
            ii.set_facecolor(scolors[3])
            ii.set_alpha(salpha)
            ii.set_zorder(3)
            

        ax1.axis([xmin,xmax,ymin,ymax]) #sets axis limits, axis does not automatically choose the best limits. syntax= ([xmin, xmax, ymin, ymax])
    
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

# =============================================================================
# #######################
# #A different plot!
# #######################
# #G for ground
# g = pd.read_csv("C:\Users\Vivien\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Coordinates and Maps\DEM_Coords.txt")
# gvalues = (g.grid_code*0.3048).values
# gpoints = np.array([tuple(x) for x in g[['x','y']].to_records(index=False)])
# 
# xmin = gpoints[:,0].min()
# xmax = gpoints[:,0].max()
# ymin = gpoints[:,1].min()
# ymax = gpoints[:,1].max()
# grid_x, grid_y = np.mgrid[xmin:xmax:1000j,ymin:ymax:1000j]
# 
# grid_z0 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'nearest')
# grid_z1 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'linear')
# grid_z2 = griddata(gpoints, gvalues, (grid_x, grid_y), method = 'cubic')
# 
# plt.figure()
# plt.subplot(311)
# plt.imshow(grid_z0.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
# plt.title('Nearest')
# plt.subplot(312)
# plt.imshow(grid_z1.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
# plt.title('Linear')
# plt.subplot(313)
# plt.imshow(grid_z2.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
# plt.title('Cubic')
# plt.show()
# 
# 
# filenames = os.listdir(gwpath)
# prefix = "WL"
# suffix = ".csv"
# files = [filename for filename in filenames if (filename.startswith(prefix) and filename.endswith(suffix))]
# files.remove("WLW11_ibp_main.csv")
# svalues = []
# spoints = []
# for f in files:
#     tpath = gwpath + f
#     sensor = f.split('_')[0]
#     print sensor
# 
#     datafile = pd.read_csv(tpath,parse_dates=['date_time'])
#     datafile = datafile[datafile['date_time'] > '2016-12-01']
#     datafile = datafile[datafile['date_time'] < '2017-05-01']
#     df_res = datafile.set_index('date_time').WS_elevation_m.resample("D").mean().to_frame().reset_index()
#     norain = pd.merge(df_res, rp) 
#     norain['raincum'] = norain.precip_cm.cumsum()/100
#     norain['WS_elev_corr'] = norain['WS_elevation_m'] - norain['raincum']
#     
#     sensor_meta = pd.read_table(gwpath+'wl_position_meta.csv',
#                                 sep=',', index_col=False,parse_dates = ['date'])
#     sensor_meta.columns = ['data_id', 'top_elev_ft', 'cable_length_ft',
#                            'ground_elev_ft', 'lat', 'long','date_change']
#     
#     pos = sensor_meta[sensor_meta['data_id'] == sensor][['long','lat']].iloc[0]
#     svalues = svalues + [norain['WS_elev_corr'].mean()]
#     spoints = spoints + [tuple(pos)]
#     
# spoints = np.array(spoints)
# svalues = np.array(svalues)
#     
# grid_w0 = griddata(spoints, svalues, (grid_x, grid_y), method = 'nearest')
# grid_w1 = griddata(spoints, svalues, (grid_x, grid_y), method = 'linear')
# grid_w2 = griddata(spoints, svalues, (grid_x, grid_y), method = 'cubic')
# 
# grid_v0 = grid_w0-grid_z0
# grid_v0[grid_v0 < 0] = 0
# 
# 
# plt.figure()
# plt.imshow(grid_w0.T, extent=(xmin,xmax,ymin,ymax), origin='lower')
# plt.title('Nearest')
# sp = pd.DataFrame(spoints)
# plt.scatter(sp[0],sp[1])
# 
# 
# sns.heatmap(grid_w0.T)
# =============================================================================
