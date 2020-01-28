# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:24:55 2018

@author: Vivien
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import cumtrapz as ct # Went with height over area as critical measure; may rigorously test later with other integrations
import re
from datetime import datetime

main_path = "C:\Users\Packman-Field\Google Drive\Packman Group\Multifunctional Urban Green Spaces Research Project\IBP Project\Documents\Road Salt\IC Analysis\\"
#file_loc = main_path + "IC Raw data\\"
local_path = "C:\Users\Packman-Field\Documents\IC_local\\"
processed_datapath = local_path + "IC Processed Data\\"
figures_datapath = local_path + "Lili_testplots\\"
file_loc = "C:\Users\Packman-Field\Documents\IC Data_Lili\IC Data\\"

filenames = os.listdir(file_loc)
allfiles = [filename for filename in filenames if filename.endswith('.txt')]

#Figure out where the data starts
def header_count(file_path):
    i=-1
    data_start_flag='min'
    temp_line=('placeholder')
    header_bottom = 0
    temp_file=open(file_path,'r') ##opens the file without loading in the data, remember to close the file once done using it.
    while (temp_line.find(data_start_flag)==i):
        temp_line=temp_file.readline()
        header_bottom = header_bottom + 1
    temp_file.close()
    return header_bottom

# Convert filenames to match Lili's format to Vivien's format
def fix_filenames(file_path):
    
    #Standards
    files = [filename for filename in allfiles if filename.startswith('Std')]
    for f in files:
        fnstrs = re.split("_|-",f) #Break old filename into strings
        conc = re.sub('[^0-9]','', fnstrs[1])
        newstring = ("Standard " + conc + " uM " + fnstrs[2]+"_"+fnstrs[3]+"-"+fnstrs[4]+"_"+fnstrs[5]+"-"+fnstrs[6])
        os.rename(file_loc+f,file_loc+newstring)

    #Bad date formats    
    files = [filename for filename in allfiles if "#" in filename]
    for f in files:
        fnstrs = f.split()
        ds = fnstrs[1].split("#")
        fnstrs[1] = "20"+ds[2]+"-"+ds[0].rjust(2,"0")+"-"+ds[1]
        newstring = " ".join(fnstrs)
        os.rename(file_loc+f,file_loc+newstring)
        
    #WL files with missing dilultion (1x)
    files = [filename for filename in allfiles if filename.startswith('WL') and "x_" not in filename]
    for f in files:
        fnstrs = f.split("_")
        fnstrs[2] = "1x_"+fnstrs[2]
        newstring = "_".join(fnstrs)
        os.rename(file_loc+f,file_loc+newstring)
        
    #WL samples with bad delimiters and backwards date format
    files = [filename for filename in allfiles if filename.startswith('WL')]
    for f in files:
        fnstrs = f.split("_")
        fnstrs[1] = fnstrs[1][-4:]+"-"+fnstrs[1][:2]+"-"+fnstrs[1][2:4]
        newstring = " ".join(fnstrs[0:3])+"_"+fnstrs[3]+"_"+fnstrs[4]
        os.rename(file_loc+f,file_loc+newstring)

# Extract data from the raw text file
# Takes the path of the text file generated by the instrument software;
# returns a data fame of time (minute) and conductivity (microsiemen/cm)
def process(temp_fn):
    fp = file_loc + temp_fn
    header_rows = header_count(fp)
    file_raw = pd.read_table(fp,skiprows = header_rows, sep = ';',names = ['time','conductivity'])
    s_name = temp_fn.split("_")[0]
    
    with open(fp,'r') as f: #With statement closes the file at the end
        lines=f.readlines() 
        timestr = lines[0]
    timestr = ' '.join(timestr.split())[:-6]
    
    runtime = datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S')
    return runtime,s_name,file_raw    

# Plots the raw timeseries (cond. vs time) on consistently sized axes
def plot_chroma(runtime,s_name,file_raw,xm = 20, ym = 12):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(file_raw.time,file_raw.conductivity)
    ax.set(title = s_name,
           xlim = [0,xm],
           ylim = [0,ym],
           xlabel = 'time (minutes)',
           ylabel = 'conducivity (microS/cm)')
    
def peakfinder(snippet):
    y = snippet.conductivity
    #x = snippet.time       # May want this later?

    #Generate statistics and find peak
    ysts = [np.mean(y),
            np.median(y),
            np.mean(y)-1.96*np.std(y)/np.sqrt(len(y)),
            np.mean(y)+1.96*np.std(y)/np.sqrt(len(y))]
    
    # Flag all rows within peak
    snippet['peak'] = False
    snippet['peak'] = np.where(snippet['conductivity']>=ysts[3], True, False)
    #snippet.index[snippet['change'] == True].tolist()

    # Identify continuous ranges where we are "in" a peak: 
    ind_peaks = []    
    ind = 0;
    tog = snippet['peak'][0]            #Toggle is the previous value
    
    for index in range(len(snippet)):
        check = snippet['peak'][index]  # Check is the current value
        if check != tog:                # When a state change occurs
            if tog == True:             # See whether we're coming off a peak and if so:
                temp_peak = [ind,index]     # Add this peak to the list of peaks 
                ind_peaks = ind_peaks + [temp_peak]
                tog = check                 # Indicate that we're now at baseline (tog = check = False)
            if tog == False:            # If a state change occurs to come ONTO a peak
                ind = index                 # Save this position as the start of the peak
                tog = check                 # Set tog = check = True (we are now in a peak)
    # Now we have a list of lists, ind_peaks, with each peak as a tuple of [start,end] indices
    
    
    # Identify elemental peaks (this is hard-coded and dependent on method; future version should incorporate 100uM single-ion standards)
    #t_F = 4.21
    t_Cl = 6.23
    #t_Nitrite = 7.47
    #t_Br = 9.52
    #t_Nitrate = 10.97

    cl_peak = []
    for peak in ind_peaks:
        p = snippet.iloc[peak[0]:peak[1]]
        pt=p.time.values.tolist()
        if pt[0] < t_Cl and pt[-1] > t_Cl:
            cl_peak = peak
    snippet.drop(['peak'],axis = 1,inplace = True)
    
    return cl_peak, np.mean(y)


# Calculate area, height, and full-width at half maximum
def calc_peakstats(file_raw,cpeak,peak_baseline):

    peak_only = file_raw[cpeak[0]:cpeak[1]].copy()          # Get the Cl peak that we found with peakfinder()
    peakmax = max(peak_only.conductivity)                   # Get the max of the peak
    halfmax = (peakmax-peak_baseline)/2+peak_baseline       # Find the half-maximum
    df_halfmax = peak_only[peak_only['conductivity'] > halfmax].reset_index(drop = True)['time'].values
    fwhm = df_halfmax[-1] - df_halfmax[0]                   # Get width at half-maximum
    
    peak_y = peak_only.conductivity.values - peak_baseline
    peak_x = peak_only.time.values
    peak_area = np.trapz(peak_y,peak_x)
    
    return peak_area,peakmax-peak_baseline,fwhm

#GMP samples
## TODO: Generalize for CBG samples (check nomenclature)
analyte = "Cl" ## Later: Iterate over all analytes (see later TODO)

## Standards; similar process to samples
def process_chroma_std(analyte, plot = True,chatty = True):
    standard_data = []
    files = [filename for filename in allfiles if filename.startswith('Standard')]
    for fn in files:
        rt,sn,fr = process(fn)
        std_conc = int(sn.split()[1])
        std_analyte = sn.split()[3]
        max_cond = max(fr.conductivity)
        rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime
        rt_fn = datetime.strftime(rt, "%Y%m%d-%H%M%S") #No punctuation runtime (for filenames)
        datarow = [std_analyte,std_conc,rt_l,max_cond]
        if(max_cond < 100):
            cp, pb = peakfinder(fr)
            if(len(cp) > 0):
                pa,pm,fwhm = calc_peakstats(fr,cp,pb)
                if chatty == True: print(sn + ": Peak area: " + str(pa))
                datarow = datarow + [pa,pm,fwhm]
            else:
                if chatty == True: print(sn + ": No " + analyte + " peak")
                datarow = datarow + [0,0,0]
            if plot == True:
                    plot_chroma(rt_fn,sn,fr)
                    snp = "_".join(sn.split())
                    snp = "-".join(snp.split("/"))
                    figname = rt_fn + "_" + snp +".png"
                    plt.savefig(figures_datapath+figname)
                    plt.close()
        else:
            if chatty == True: print(sn + ": No peaks")
            datarow = datarow + [np.nan,np.nan,np.nan]
        standard_data = standard_data + [datarow]

    # Create dataframe for analyte standards
    df_standards = pd.DataFrame(standard_data)
    df_standards.columns = ['analyte','conc_uM','analysis_time','max_cond','peak_area','peak_height','fwhm']
    df_standards.analysis_time=pd.to_datetime(df_standards.analysis_time) #Includes 100um Single-anion stds (not needed for calibration but good to have as evidence of peak timing)
    lin_stds = df_standards[df_standards['analyte'] == 'mixedanion'].copy().sort_values(["analysis_time","conc_uM"])
    df_standards = lin_stds.sort_values('conc_uM')
    return df_standards

def process_chromatogram(analyte,plot = True,chatty = True):
    analysis_data = []
    files = [filename for filename in allfiles if filename.startswith('WL')]
    for fn in files:
        rt,sn,fr = process(fn)  # Get data frame
        s_id = sn.split()[0]    # Get identifying metadata: sample ID, runtime, dilution
        s_date = sn.split()[1]
        rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime
        rt_fn = datetime.strftime(rt, "%Y%m%d-%H%M%S") #No punctuation runtime (for filenames)
        dil = int(sn.split()[2].strip('x'))
        max_cond = max(fr.conductivity)     # Identify the max conductivity to decide whether or not to analyze peaks (useful when there is instrument error)
        datarow = [s_id,s_date,rt_l,dil,max_cond]  # Include metadata in data row
        if(max_cond < 100):                          # If max conductivity reasonable:
            cp, pb = peakfinder(fr)       # Find analyte peak
            if(len(cp) > 0):                         # If analyte peak exists
                pa, pm, fwhm = calc_peakstats(fr,cp,pb)           # Calculate area, height at max, full width at half max
                if chatty == True: print(sn + ": Peak area: " + str(pa))       # Status report
                datarow = datarow + [pa,pm,fwhm]            # Add data row to data file
                if plot == True: plot_chroma(rt_fn,sn,fr)                       # Plot on default axes
            else:                                       # If no analyte peak:
                if chatty == True: print(sn + ": No " + analyte + " peak")     # status report
                datarow = datarow + [0,0,0]                 # All zeroes (no detectable analyte, but data is good)
        else:                                       # If max conductivity is excessive:
            if chatty == True: print(sn + ": No peaks")                    # Status report
            datarow = datarow + [np.nan,np.nan,np.nan]  # All not-a-number (data is bad)
        analysis_data = analysis_data + [datarow]   # Add result of chromatogram analyses to data frame
        snp = "_".join(sn.split())
        snp = "-".join(snp.split("/"))
        figname = rt_fn + "_" + snp +".png"
        plt.savefig(figures_datapath+figname)
        plt.close()
        
    df_data = pd.DataFrame(analysis_data)       # Format dataframe
    df_data.columns = ['name','sample_date','analysis_time','dilution','max_cond','peak_area','peak_height','fwhm']
    df_data.analysis_time=pd.to_datetime(df_data.analysis_time)
    return df_data

## TODO: Calibrate based on runtime (daily std runs correspond to samples analyzed that day)

#formulae used:
#area = conc * fit[0] + fit[1]
#conc = (area - fit[1])/fit[0]

df_st = process_chroma_std("Cl")
df_d = process_chromatogram("Cl")
fit = np.polyfit(df_st.conc_uM,df_st.peak_area,1) ## Todo: Separate standard runs
df_d['conc_uM'] = ((df_d['peak_area'] - fit[1])/fit[0])*df_d['dilution']

# Get only data for good runs
## TODO: Move this to the end and publish as full data table
data_reduced = df_d[['name','sample_date','conc_uM']].dropna()

#Save data
df_d.to_csv(processed_datapath + "IC_2019_sampledata_detailed.csv",index = False)
df_st.to_csv(processed_datapath + "IC_2019_standards.csv",index = False)
data_reduced.to_csv(processed_datapath + "IC_2019_sampledata_reduced.csv",index = False)

## TODO: Set up runs for all 5 analytes and summarize into one data table

#df_data.sort_values('analysis_time')
#df_data.sort_values(['dilution','analysis_time'])
#df_data[df_data.max_cond > 10]

# =============================================================================
# temp_fn = 'WLW8 3#27#18 10x_Warta-PC_20180420-162626.txt' 
# rt,sn,fr = process(temp_fn)
# plot_chroma(rt,sn,fr)
# 
# =============================================================================