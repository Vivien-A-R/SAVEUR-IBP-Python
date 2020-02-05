# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:24:55 2018

Instructions for use:
    1. Create data directory that will contain both raw and processed data
        main directory > site name > "IC Data" 
        main directory > site name > "Processed"
        main directory > site name > "Processed" > "datafiles"
        main directory > site name > "Processed" > "figures"
    2. Set variable local_path to the path to the folder containing the site folders
    3. Set variable temp_fs to the sitename EXACTLY matching the site name in the file path
    4. Run set_fs()
    5. Run fix_filenames() if filenames don't match the format: [sample id]_Warta-PC_[timestamp].txt
    6. Set up analyte list:
        potential_analytes = ["F",
                              "Cl",
                              "Br",
                              "Phosphate",
                              "Sulfate",
                              "Nitrate",
                              "Nitrite"]
    7. Run full_run() with your analyte list as the only argument to process with reports and without plotting
       Other args are, in order: Report Standards, Plot Standards, Report Samples, Plot Samples
       Example run:
           full_run(["F"],False, False, False, True)
           
           Results in processing all samples for F from the site selected in set_fs(),
               not printing real-time results in the console,
               but quietly plotting and saving figures of sample data only.
           Do note that even for a single analyte, the function requires a list, not a string.

@author: Vivien
"""

import pandas as pd
import os as os
import matplotlib.pyplot as plt
import numpy as np
#from scipy.integrate import cumtrapz as ct # Went with height over area as critical measure; may rigorously test later with other integrations
import re
from datetime import datetime

## Set these!
##########################
local_path = "C:\Users\Packman-Field\Documents\IC Data\\"

temp_fs = "CBG"

def set_fs(fieldsite_name):
    raw_path = local_path + fieldsite_name + "\\IC Data\\"
    proc_path = local_path + fieldsite_name + "\\Processed\\"
    return raw_path, proc_path

file_loc, processed_loc = set_fs(temp_fs)
filenames = os.listdir(file_loc)
allfiles = [filename for filename in filenames if filename.endswith('.txt')]

#potential_analytes = ["F","Cl","Br","Phosphate","Sulfate","Nitrate"]
potential_analytes = ["Phosphate","Sulfate","Nitrate"]

#############################

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
# For new sample sets, will need to make a new correction rule
# ONLY RUN ONCE.
def fix_filenames():
    
    #Standards
    # Standard [conc] uM [analyte]_Water-PC_[timestamp].txt
    # Example: Standard 100 uM mixedanion_Warta-PC_20191011-103847
    files = [filename for filename in allfiles if filename.startswith('Std')]
    for f in files:
        fnstrs = re.split("_|-",f) #Break old filename into strings
        conc = re.sub('[^0-9]','', fnstrs[1])
        newstring = ("Standard " + conc + " uM " + fnstrs[2]+"_"+fnstrs[3]+"-"+fnstrs[4]+"_"+fnstrs[5]+"-"+fnstrs[6])
        os.rename(file_loc+f,file_loc+newstring)

    # WL Sample format:
    # [sensor id] [sample date] [dilution]_Warta-PC_['timestamp'].txt
    # Example: WLW2 2018-05-29 1x_Warta-PC_20190917-135918.txt

    #Bad date formats
    # Only for WL samples, IC software converts / to # in date formats; also rearrange date to YYYY-MM-DD    
    files = [filename for filename in allfiles if "#" in filename]
    for f in files:
        fnstrs = f.split()
        ds = fnstrs[1].split("#")
        fnstrs[1] = "20"+ds[2]+"-"+ds[0].rjust(2,"0")+"-"+ds[1]
        newstring = " ".join(fnstrs)
        os.rename(file_loc+f,file_loc+newstring)
        
    #WL files with missing dilultion (1x)
    # Only for GMP files; assume others are not diluted or that dilution is elsewhere
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
        
    # CBG samples
    # [sample id]_Warta-PC_[timestamp].txt
    # Example: S3-10-storm11_Warta-PC_20191025-075649.txt
    files = [filename for filename in allfiles if filename.startswith('S') and "uM" not in filename]
    for f in files:
        fnstrs = f.split("_")
        newstring = "-".join(fnstrs[0:3])+"_"+"_".join(fnstrs[3:5])
        print newstring
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
           ylabel = 'conductivity (microS/cm)')

# "Switch" for selecting time based on arg
def analyte_time_set(ion):
    switcher = {
        "F" : 4.21 ,
        "Cl": 6.23,
        "Nitrite": 7.47,
        "Phosphate": 9.24,
        "Br": 9.52,
        "Nitrate": 10.97,
        "Sulfate": 10.81
    }
    return switcher.get(ion)


#Tolerance is because peaks seem to shift a bit between runs, which makes some get skipped; play with this if you're concerned about peak overlap
## TODO: Concerning bit: Nitrite peak is too small for conventional algorithm (points that differ from the mean within 95% CI are in a peak); need to do something different for nitrate/nitrate
## TODO: Only process small chunks to capture analytes with relatively small peaks (fix for nitrate)
def peakfinder(snippet,analyte,tolerance = 0.1):
    y = snippet.conductivity
    #x = snippet.time       # May want this later?

    #Generate statistics and find peaks
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
    
    
    t_expected = analyte_time_set(analyte)
    
    ion_peak = []
    for peak in ind_peaks:
        p = snippet.iloc[peak[0]:peak[1]]
        pt=p.time.values.tolist()
        if pt[0] < t_expected+tolerance and pt[-1] > t_expected-tolerance:
            ion_peak = peak
    snippet.drop(['peak'],axis = 1,inplace = True)
    
    return ion_peak, np.mean(y)

# Calculate area, height, and full-width at half maximum
def calc_peakstats(file_raw,ipeak,peak_baseline):

    peak_only = file_raw[ipeak[0]:ipeak[1]].copy()          # Get the analyte ion peak that we found with peakfinder()
    peakmax = max(peak_only.conductivity)                   # Get the max of the peak
    halfmax = (peakmax-peak_baseline)/2+peak_baseline       # Find the half-maximum
    df_halfmax = peak_only[peak_only['conductivity'] > halfmax].reset_index(drop = True)['time'].values
    fwhm = df_halfmax[-1] - df_halfmax[0]                   # Get width at half-maximum
    
    peak_y = peak_only.conductivity.values - peak_baseline
    peak_x = peak_only.time.values
    peak_area = np.trapz(peak_y,peak_x)
    
    return peak_area,peakmax-peak_baseline,fwhm

## Standards; similar process to samples
# Options for analyte : ["F","Cl","Nitrite","Br","Nitrate"], case sensitive, misspellings will throw error
def process_chroma_std(analyte, sitename = "GMP", plot = True,chatty = True):
    standard_data = []
    files = [filename for filename in allfiles if filename.startswith('Standard')]
    print("Processing " + sitename + " standards for " + analyte)
    for fn in files:
        rt,sn,fr = process(fn)
        std_conc = int(sn.split()[1])
        std_analyte = sn.split()[3]
        max_cond = max(fr.conductivity)
        rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime
        rt_fn = datetime.strftime(rt, "%Y%m%d-%H%M%S") #No punctuation runtime (for filenames)
        datarow = [std_analyte,std_conc,rt_l,max_cond]
        if(max_cond < 100):
            cp, pb = peakfinder(fr,analyte)
            if(len(cp) > 0):
                pa,pm,fwhm = calc_peakstats(fr,cp,pb)
                if chatty == True: print(sn + ": " + analyte + " peak area: " + str(pa))
                datarow = datarow + [pa,pm,fwhm]
            else:
                if chatty == True: print(sn + ": No " + analyte + " peak")
                datarow = datarow + [0,0,0]
            if plot == True:
                    plot_chroma(rt_fn,sn,fr)
                    snp = "_".join(sn.split())
                    snp = "-".join(snp.split("/"))
                    figname = rt_fn + "_" + snp +".png"
                    plt.savefig(processed_loc + "figures//" +figname)
                    plt.close()
        else:
            if chatty == True: print(sn + ": No peaks")
            datarow = datarow + [np.nan,np.nan,np.nan]
        standard_data = standard_data + [datarow]

    # Create dataframe for analyte standards
    df_standards = pd.DataFrame(standard_data)
    df_standards.columns = ['analyte','conc_uM','analysis_time','max_cond','peak_area','peak_height','fwhm']
    df_standards.analysis_time=pd.to_datetime(df_standards.analysis_time) #Includes 100um Single-anion stds (not needed for calibration but good to have as evidence of peak timing)
    df_standards = df_standards[df_standards['analyte'] == 'mixedanion'].sort_values(["conc_uM","analysis_time"])
    return df_standards

#Sitename options are presently ["GMP","CBG"], must match the folder names in the IC Data directory
def process_chromatogram(analyte,sitename = "GMP" ,plot = True,chatty = True):
    analysis_data = []
    files = allfiles
    if sitename == "GMP": files = [filename for filename in allfiles if filename.startswith('WL')]
    elif sitename == "CBG": files = [filename for filename in allfiles if filename.startswith('S') and "uM" not in filename]
    else: files = allfiles
    
    print("Processing " + sitename+ " samples for " + analyte)
    
    for fn in files:
        rt,sn,fr = process(fn)  # Get data frame
        s_id = sn.split()[0]    # Get identifying metadata: sample ID, runtime, dilution
        max_cond = max(fr.conductivity)     # Identify the max conductivity to decide whether or not to analyze peaks (useful when there is instrument error)
        if sitename == "GMP":
            s_date = sn.split()[1]
            dil = int(sn.split()[2].strip('x'))
        if sitename == "CBG":
            dil = 1.0
        rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime
        rt_fn = datetime.strftime(rt, "%Y%m%d-%H%M%S") #No punctuation runtime (for filenames)
        if sitename == "GMP": datarow = [s_id,s_date,rt_l,dil,max_cond]  # Include metadata in data row
        else: datarow = [s_id,rt_l,dil,max_cond]
            
        if(max_cond < 100):                          # If max conductivity reasonable:
            cp, pb = peakfinder(fr,analyte)       # Find analyte peak
            if(len(cp) > 0):                         # If analyte peak exists
                pa, pm, fwhm = calc_peakstats(fr,cp,pb)           # Calculate area, height at max, full width at half max
                if chatty == True: print(sn + ": " + analyte + " peak area: " + str(pa))       # Status report
                datarow = datarow + [pa,pm,fwhm]            # Add data row to data file
                     # Plot on default axes
            else:                                       # If no analyte peak:
                if chatty == True: print(sn + ": No " + analyte + " peak")     # status report
                datarow = datarow + [0,0,0]                 # All zeroes (no detectable analyte, but data is good)
        else:                                       # If max conductivity is excessive:
            if chatty == True: print(sn + ": No peaks")                    # Status report
            datarow = datarow + [np.nan,np.nan,np.nan]  # All not-a-number (data is bad)
        analysis_data = analysis_data + [datarow]   # Add result of chromatogram analyses to data frame

        if plot == True:
            plot_chroma(rt_fn,sn,fr)  
            snp = "_".join(sn.split())
            snp = "-".join(snp.split("/"))
            figname = rt_fn + "_" + snp +".png"
            plt.savefig(processed_loc + "figures//" +figname)
            plt.close()
        
    df_data = pd.DataFrame(analysis_data)       # Format dataframe

    if sitename == "GMP": df_data.columns = ['name','sample_date','analysis_time','dilution','max_cond','peak_area','peak_height','fwhm']
    else: df_data.columns = ['name','analysis_time','dilution','max_cond','peak_area','peak_height','fwhm']

    df_data.analysis_time=pd.to_datetime(df_data.analysis_time)
    df_data.sort_values("analysis_time",inplace = True)
    return df_data


## TODO: Drop std run days with bad data (entire day)
def full_run(selected_analytes,st_verbose=True,st_figures=False,d_verbose = True,d_figures=False):
    for temp_analyte in selected_analytes:
    
        #formulae used:
        #area = conc * fit[0] + fit[1]
        #conc = (area - fit[1])/fit[0]
        
        df_st = process_chroma_std(temp_analyte,temp_fs,chatty = st_verbose, plot = st_figures)
        df_d = process_chromatogram(temp_analyte,temp_fs,chatty = d_verbose,plot = d_figures)
    
        fit = np.polyfit(df_st.conc_uM,df_st.peak_area,1)       # Regress standard conc w/ chromatogram peak area
        #reg_concs = df_st.conc_uM.unique().tolist()
        #reg_means = 
        #reg_stdevs = 
        # Make this into a data table and use to drop outliers
        
        df_d['conc_uM'] = ((df_d['peak_area'] - fit[1])/fit[0])*df_d['dilution']    # Use regression to calculate concentration from sample chromatogram peak area
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        lin_x_l = 0
        lin_x_r = 100
        lin_y_l = lin_x_l * fit[0] + fit[1]
        lin_y_r = lin_x_r * fit[0] + fit[1]
        s_title = "Standard curve for "+ temp_analyte
        figname = temp_analyte + "_std_all.png"
        
        ax.plot(df_st.conc_uM, df_st.peak_area,'.')
        ax.plot([lin_x_l, lin_x_r], [lin_y_l, lin_y_r])
        ax.set(title = s_title,
               ylabel = 'peak_area (uS/cm*min) ',
               xlabel = 'concentration (uM)')
        plt.savefig(processed_loc + "figures//" +figname)
        plt.close()
        
        df_d.sort_values("analysis_time",inplace = True)
        
        #Save data
        df_d.to_csv(processed_loc + "datafiles//" + "IC_2019_" + temp_analyte + "_sampledata_detailed.csv",index = False)
        df_st.to_csv(processed_loc + "datafiles//" + "IC_2019_" + temp_analyte + "_standards.csv",index = False)
        return df_d,df_st
        
# Get only data for good runs
## TODO: Move this to the end and publish as full data table w/ all analytes
#data_reduced = df_d[['name','sample_date','conc_uM']].dropna()
#data_reduced.to_csv(processed_datapath + "IC_2019_sampledata_reduced.csv",index = False)