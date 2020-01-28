# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:54:11 2017

@author: Vivien
"""
from __future__ import division
import numpy as np #pythons numerical package
import pandas as pd #pythons data/timeseries package
import os as os #data in/out package
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import time
import datetime
import matplotlib as mpl
from meta_get import meta_get

mpl.rcParams.update({'font.size': 14})


#Plotting only, no spectral

data_path = 'C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Processed Water Level Data\\'

#pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

#Sensor/well metadata (for sample elevations)
sensor_meta = meta_get(data_path)
sensor_meta['path'] = data_path+sensor_meta['data_id']+"_ibp_main.csv"

def plot_raw(sensor_id = "all"):
    if(sensor_id == "all"):
        print("Default settings, loop through all sensors.")
        baro_plot()
        #Iterate through all sensors and plot them all on separate graphs with default axes
        for index,row in sensor_meta.iterrows():
            print row['data_id']
            datafile = pd.read_csv(row['path'],parse_dates=['date_time'])

            fig, ax = sep_plot(datafile,True)
            #presstemp_plot(datafile,row['data_id'])
            fig.suptitle(row['data_id'])
            bottomelv = (row['top_elev_ft']-row['cable_length_ft'])*0.3048
            gelv = (row['ground_elev_ft'])*0.3048
            plt.axhline(y=bottomelv,c='r')
            plt.axhline(y=gelv,c='g')

        multipage("ind_plots_all")

    elif(not sensor_meta.data_id.str.contains(sensor_id).any()):
        print("There is no data file for this sensor!")
        #Stops here!

    else:
        print("Generate plot for sensor "+sensor_id+" only.")
        datafile = pd.read_csv(sensor_meta[sensor_meta.data_id == sensor_id].path.drop_duplicates().item(),parse_dates=['date_time'])
        fig,ax = sep_plot(datafile,True)
        fig.suptitle(sensor_id)
    
    return datafile
# =============================================================================
#         topelev_ft = sensor_meta[sensor_meta.data_id == sensor_id].top_elev_ft.item()
#         cablelen_ft = sensor_meta[sensor_meta.data_id == sensor_id].cable_length_ft.item()
#         gelv = sensor_meta[sensor_meta.data_id ==  sensor_id].ground_elev_ft.item()
#         plt.axhline(y=(topelev_ft-cablelen_ft)*0.3048,c='r',label = 'Sensor Elevation')
#         plt.axhline(y=(gelv)*0.3048,c='g',label = 'Ground Surface')
# =============================================================================


#Plot a single timeseries on its own axes, autoset x and y range
def sep_plot(datafile,skip_qc=False):
    fig,ax = plt.subplots(figsize = (12,8))
    if(skip_qc == True):datafile.loc[datafile.qual_c<1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
    ax.plot(datafile.date_time,datafile.WS_elevation_m,label = 'Water Level') #Shows the original signal
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    plt.ylabel('Water table elevation (meters)')
    plt.xlabel('Date-time (minor ticks = days)')
    fig.autofmt_xdate()
    return fig,ax

#Plots on stacked axes, shared x-axis
def stack_plot(sets= (("WLW2","WLW3","WLW4","WLW5"),("WLS1","WLS2","WLS3","WLS4"),
            ("WLS5","WLS6","WLS7","WLS8")),stitch = False):
    paths = sensor_meta[['data_id','path']].drop_duplicates()
    level_data = dict((name,[]) for name in paths)

    for index,row in paths.iterrows():
        datafile = pd.read_csv(row['path'],parse_dates = ['date_time'])
        datafile.loc[datafile.qual_c<1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
        level_data[row['data_id']]=datafile

    for set in sets:
        fig = plt.figure()
        i = 1
        for ts in set:
            timestamps = level_data[ts].date_time
            ax = plt.subplot(len(set),1,i)
            ax.ticklabel_format(useOffset=False)

            #Brute-forcing axis limits.
            #Please forgive me.
            elmin = 183.5
            elmax = 185.5
            ax.set_ylim(elmin,elmax)
            ax.set_yticks((184.0,185))
            minor_locator = AutoMinorLocator(2)
            ax.yaxis.set_minor_locator(minor_locator)

            #days = mdates.DayLocator()
            months = mdates.MonthLocator(interval = 2)
            halfmonths = mdates.MonthLocator()
            datemin = datetime.date(2016,7,1)
            datemax = datetime.date(2017,11,1)
            ax.set_xlim(datemin, datemax)
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_minor_locator(halfmonths)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%B %Y'))

            ax.plot(timestamps,level_data[ts].WS_elevation_m,label = 'Water Level')

            #bottomelv = (sensor_meta[sensor_meta.data_id==ts]['top_elev_ft'].item()-sensor_meta[sensor_meta.data_id==ts]['cable_length_ft'].item())*0.3048
            gelv = sensor_meta[sensor_meta.data_id==ts]['ground_elev_ft'].drop_duplicates().item()*0.3048
            #plt.axhline(y=bottomelv,c='r')
            plt.axhline(y=gelv,c='g',label = 'Ground Surface')
            #ax.text(datemin+3, elmax-0.5,ts)

            print ts
            i = i+1
        fig.suptitle(", ".join(set), y=0.93)
        #plt.ylabel('Water table elevation (meters)')
        fig.autofmt_xdate()
        plt.subplots_adjust(hspace=0.0001)
        #plt.legend()
    if(stitch == True): multipage("stack_plots")


#Meh.
def ols_plot(set_window,stitch = False):
    mols = pd.stats.ols.MovingOLS

    for index,row in sensor_meta.iterrows():
        datafile = pd.read_csv(row['path'],index_col = 0)

        model = mols(x=datafile.run_time,
                     y=datafile.WS_elevation_m,
                     window_type='rolling',
                     window=set_window,
                     intercept=True)
        datafile['Y_hat'] = model.y_predict

        plt.figure()
        plt.plot(datafile.run_time,datafile.WS_elevation_m,c='r')
        plt.plot(datafile.run_time,datafile.Y_hat,c='b')
        plt.suptitle(row['data_id'])
    if(stitch == True): multipage("ols_plots_all")

#Splits the time series along calendar seasons
def season_split(datafile):
    df_season = datafile.copy()
    df_season['date_strip'] = df_season.date_time.dt.month.map("{:02}".format) + df_season.date_time.dt.day.map("{:02}".format)

    sspring = "0322"
    ssummer = "0622"
    swinter = "1222"
    sautumn = "0922"

    df_spring = df_season[(df_season['date_strip'] > sspring) & (df_season['date_strip'] < ssummer)].drop(['date_strip'],1)
    df_summer = df_season[(df_season['date_strip'] > ssummer) & (df_season['date_strip'] < sautumn)].drop(['date_strip'],1)
    df_fall = df_season[(df_season['date_strip'] > sautumn) & (df_season['date_strip'] < swinter)].drop(['date_strip'],1)
    df_winter = df_season[(df_season['date_strip'] > swinter) ^ (df_season['date_strip'] < sspring)].drop(['date_strip'],1)

    return df_spring,df_summer,df_fall,df_winter

#Does calculations for histograms; skipping qc-flagged values
def hist_calc(datafile,norm = False,plotit = False,bins = None):
    print "Calculate histogram"
    datafile.loc[datafile.qual_c<1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
    a = datafile.WS_elevation_m.as_matrix()
    print "Total number of records:"
    print np.size(a)
    a = a[~np.isnan(a)]
    print "Number of non-NaN records used for calculation:"
    print np.size(a)

    bins = np.linspace(183,186.5,43)
    hist,bins = np.histogram(a,bins = bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    if(norm == True): hist = hist/hist.sum()
    if(plotit == True):
        plt.bar(center, hist, align='center', width=width)
        plt.show()
    else:
        return hist, bins, width, center, np.size(a)

#Plots seasonal histograms
def hist_seasonal(datafile,norm = False,bins = None):
    spring, summer, fall, winter = season_split(datafile)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True)

    h,b,w,c,n = hist_calc(spring)
    if(norm == True): h=h/h.sum()
    ax1.bar(c,h,align = 'center',width = w,color = 'g',label='Spring',edgecolor = "none")
    ax1.legend(frameon=False,handlelength=0, handletextpad=0)
    h,b,w,c,n = hist_calc(summer)
    if(norm == True): h=h/h.sum()
    ax2.bar(c,h,align = 'center',width = w,color = 'k',label='Summer',edgecolor = "none")
    ax2.legend(frameon=False,handlelength=0, handletextpad=0)
    h,b,w,c,n = hist_calc(fall)
    if(norm == True): h=h/h.sum()
    ax3.bar(c,h,align = 'center',width = w,color = 'r',label='Fall',edgecolor = "none")
    ax3.legend(frameon=False,handlelength=0, handletextpad=0)
    h,b,w,c,n = hist_calc(winter)
    if(norm == True): h=h/h.sum()
    ax4.bar(c,h,align = 'center',width = w,color = 'b',label='Winter',edgecolor = "none")
    ax4.legend(frameon=False,handlelength=0, handletextpad=0)
    # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    #f.suptitle("Seasonally-split normalized histograms")

#Plots overall and seasonal histograms for all sensors
def hist_all():
    for index,row in sensor_meta.iterrows():
        print row['path']
        datafile = pd.read_csv(row['path'],parse_dates = ["date_time"])
        fig = plt.figure()
        hist_calc(datafile,True,True)
        bottomelv = (row['top_elev_ft']-row['cable_length_ft'])*0.3048
        gelv = (row['ground_elev_ft'])*0.3048
        plt.axvline(x=bottomelv,c='r')
        plt.axvline(x=gelv,c='g')
        fig.suptitle(row['data_id'] + ' Normalized, Annual Histogram',y=0.93)

        hist_seasonal(datafile,True)
        plt.suptitle(row['data_id'] + ' Normalized, Seasonal Histogram',y=0.93)

    multipage('hist_plots')

def averaging(skip_qc = False):
    ind = ['data_id','average_annual','numobs_annual',
           'average_spring','numobs_spring',
           'average_summer','numobs_summer',
           'average_fall','numobs_fall',
           'average_winter','numobs_winter']

    avglevels = pd.DataFrame()

    for index,row in sensor_meta.iterrows():
        templevels = pd.Series()

        templevels = templevels.append(pd.Series(row['data_id']))
        datafile = pd.read_csv(row['path'],parse_dates = ["date_time"])

        if(skip_qc == True):
            datafile.loc[datafile.qual_c==0,"WS_elevation_m"]=np.nan #Skip qc-flagged values
            a1 = datafile.WS_elevation_m.as_matrix()
            a = a1[~np.isnan(a1)]
        else:
            a = datafile.WS_elevation_m.as_matrix()

        avg = np.average(a)
        n = np.size(a)
        templevels = pd.concat([templevels,pd.Series(avg),pd.Series(n)])

        seasons = season_split(datafile)
        for season in seasons:
            if(skip_qc == True):
                season.loc[season.qual_c < 1,"WS_elevation_m"]=np.nan #Skip qc-flagged values
                a1 = season.WS_elevation_m.as_matrix()
                a = a1[~np.isnan(a1)]
            else:
                a = season.WS_elevation_m.as_matrix()

            avg = np.average(a)
            n = np.size(a)
            templevels = pd.concat([templevels,pd.Series(avg),pd.Series(n)])

        avglevels = pd.concat([avglevels,templevels],axis=1)

    avglevels = avglevels.transpose()
    avglevels.columns = ind
    avglevels.reset_index()
    return avglevels
    #avglevels.to_csv("averages.csv",sep=',',index=False)

#Plot raw temp and pressure signal
def presstemp_plot(datafile,sid):

    x_time = pd.to_datetime(datafile.date_time)

    fig, ax1 = plt.subplots()
    #ax1.set_xticks(np.arange(0,datafile.run_time.iloc[-1]/(24*60*60),14))
    #ax1.set_xticks(np.arange(0,datafile.run_time.iloc[-1]/(24*60*60),7),minor = True)

    days = mdates.DayLocator()
    months = mdates.MonthLocator()
    datemin = datetime.date(2016,7,1)
    datemax = datetime.date(2017,5,1)
    ax1.set_xlim(datemin, datemax)
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_minor_locator(days)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax1.plot(x_time,datafile.pressure_pa,'b-')
    ax1.set_ylabel('Pressure (Pa)', color='b')
    ax2 = ax1.twinx()
    ax2.plot(x_time,datafile.temperature_c,'r-')
    ax2.set_ylabel('Temperature',color='r')
    plt.xlabel('Date/time')
    plt.suptitle(sid)
    fig.autofmt_xdate()

def baro_plot():
    datafile = pd.read_csv(data_path + '\\baro_ibp_main.csv',parse_dates = ['date_time'])
    fig = plt.figure()
    plt.plot(datafile.date_time,datafile.pressure_pa)
    plt.ylabel('Pressure (pa)')
    plt.xlabel('Date/time')
    plt.suptitle("Baro")
    fig.autofmt_xdate()

#Plots on same axes
#Modified for short time series to show precipitation
def comb_plot(sets = (("WLS1","WLS8","WLS3","WLS4"),
            ("WLS5","WLS6","WLS7","WLS8"),
            ("WLW1","WLW2","WLW3"),
            ("WLW4","WLW9","WLW8","WLW7"),
            ("WLW5","WLW6","WLW10")) ,stitch = False):
    level_data = dict((name,[]) for name in sensor_meta.data_id)
    start = datetime.datetime.strptime('2016-07-10 00:00:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime('2016-07-30 00:00:00', '%Y-%m-%d %H:%M:%S')

    for index,row in sensor_meta.iterrows():
            datafile = pd.read_csv(row['path'],index_col = 0,parse_dates = ['date_time'])
            level_data[row['data_id']]=datafile

    for set in sets:
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        j=0
        for ts in set:
            level_data[ts] = level_data[ts][level_data[ts].date_time < end]
            #df_smp = df_smp[df_smp.date_time > start]
            ax1.plot(level_data[ts].date_time,level_data[ts].WS_elevation_m,linestyle = ['-','--'][j],color = 'black',label = ts)
            j = j+1
        fig.autofmt_xdate()
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)
        ax1.set_yticks((184,184.5,185))
        precip = pd.read_csv('C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\data_products\precip.csv',sep = ',')
        precip.columns = ['date_time', 'precip_30min_in']
        precip.date_time = pd.to_datetime(precip.date_time)
        ax2.plot(precip.date_time, precip.precip_30min_in,label = "Precipitation (inches)")
        ax2.set_yticks((0,0.2,0.4))
        #plt.legend()
        #ax2.text(textx,0.5, "Precipitation (inches)",color='royalblue')
        myFmt = mdates.DateFormatter('%B %d')
        ax1.xaxis.set_major_formatter(myFmt)
        plt.subplots_adjust(hspace=0.0001)

    if(stitch == True): multipage('comb_plots')

def comb_plot2(set1,set2,stitch = False):
    level_data = dict((name,[]) for name in sensor_meta.data_id)
    #start = datetime.datetime.strptime('2016-07-10 00:00:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime('2016-07-30 00:00:00', '%Y-%m-%d %H:%M:%S')

    for index,row in sensor_meta.iterrows():
            datafile = pd.read_csv(row['path'],index_col = 0)
            level_data[row['data_id']]=datafile


    fig,(ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize = (12,6))
    j=0
    for ts in set1:
        level_data[ts].date_time = pd.to_datetime(level_data[ts].date_time)
        level_data[ts] = level_data[ts][level_data[ts].date_time < end]
        #df_smp = df_smp[df_smp.date_time > start]
        ax1.plot(level_data[ts].date_time,level_data[ts].WS_elevation_m,linestyle = ['-','--'][j],color = 'black',label = ts)
        j = j+1
    j=0
    for ts in set2:
        level_data[ts].date_time = pd.to_datetime(level_data[ts].date_time)
        level_data[ts] = level_data[ts][level_data[ts].date_time < end]
        #df_smp = df_smp[df_smp.date_time > start]
        ax2.plot(level_data[ts].date_time,level_data[ts].WS_elevation_m,linestyle = ['-','--'][j],color = 'black',label = ts)
        j = j+1

    #fig.autofmt_xdate()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=4, mode="expand", borderaxespad=0.)

    ax1.set_ylim(ymin = 183.8, ymax = 185.0)
    ax2.set_ylim(ymin = 183.8, ymax = 185.0)
    ax1.set_yticks((184.0,184.2,184.4,184.6,184.8))
    ax2.set_yticks((184.0,184.2,184.4,184.6,184.8))

    precip = pd.read_csv('C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\data_products\precip.csv',sep = ',')
    precip.columns = ['date_time', 'precip_30min_in']
    precip.date_time = pd.to_datetime(precip.date_time)
    ax3.plot(precip.date_time, precip.precip_30min_in,label = "Precipitation (inches)")
    ax3.set_yticks((0,0.2,0.4,0.6))
    #plt.legend()
    #ax2.text(textx,0.5, "Precipitation (inches)",color='royalblue')
    myFmt = mdates.DateFormatter('%B %d')
    ax1.xaxis.set_major_formatter(myFmt)
    plt.subplots_adjust(hspace=0.0001)

    if(stitch == True): multipage('comb_plots')

#Turn this into something to turn these into PNG and THEN into a PDF
#Pdf stitcher
#Save open figures
def multipage(filename, figs=None, dpi=200):
    print "Stitching PDF"
    filename = filename +'_' + time.strftime("%Y%m%d")+".pdf"
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    plt.close("all")
    os.startfile(filename)