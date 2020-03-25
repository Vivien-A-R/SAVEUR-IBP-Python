# -*- coding: utf-8 -*-
"""
Ion chomatography processing code - Vivien Rivera 2020
Instructions for use:
    1. Create data directory that will contain both raw and processed data
        main directory > site name > "IC Data" 
        main directory > site name > "Processed"
        main directory > site name > "Processed" > "datafiles"
        main directory > site name > "Processed" > "figures"
    2. Set variable local_path to the path to the folder containing the site folders
    3. Set variable temp_fs to the sitename EXACTLY matching the site name in the file path
    4. Run this entire python script to set up variables and functions
    5. Run fix_filenames() if filenames don't match the format: [sample id]_Warta-PC_[timestamp].txt
        RUN THIS ONLY ONCE; it's not a smart function.
    6. Run full_run() with your analyte list as the only argument to process with reports and default to dimension = "peak_area" for regression
       Example run:
           full_run(['F','Br'],'peak_height',False])
           
           Results in processing all samples for F and Br from the field site set with temp_fs
               not printing real-time results in the console,
               and calculating the regression using peak height instead of peak area
           Do note that even for a single analyte, the function requires a list, not a string.
           Running full_run() with no arguments processes all samples for all analytes,
               reporting real-time results in the console,
               and calculating regression with peak area
    7. Figures are not generate by the full_run() function. Use plot_all() with no
        parameters to generate plots of the chromatographs for named samples that match
        the naming schemes established in the fix_filenames function. The same function
        including the argument stds = True to plot all chromatographs in the folder identified as standards
"""
import pandas as pd
import os as os
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from scipy import stats
#from scipy.integrate import cumtrapz as ct # Went with height over area as critical measure; may rigorously test later with other integrations
import re
from datetime import datetime

## Set these!
##########################
local_path = "C:\Users\Packman-Field\Documents\IC Data\\"

temp_fs = "GMP"
potential_analytes = ["F","Cl","Br","Phosphate","Sulfate","Nitrate","Nitrite"]

file_loc = local_path + temp_fs + "\\IC Data\\"
processed_loc = local_path + temp_fs + "\\Processed\\"

filenames = os.listdir(file_loc)
allfiles = [filename for filename in filenames if filename.endswith('.txt')]

#############################
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

# Get the list of files to work with
def get_runfiles(stds = False):
    if stds == True: files = [filename for filename in allfiles if filename.startswith('Standard')]
    elif temp_fs == "GMP": files = [filename for filename in allfiles if filename.startswith('WL')]
    elif temp_fs == "CBG": files = [filename for filename in allfiles if filename.startswith('S') and "uM" not in filename]
    else: files = allfiles
    return files

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
def plot_chroma(runtime,s_name,file_raw,xm = 20):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(file_raw.time,file_raw.conductivity)
    ax.set(title = s_name,
           xlim = [0,xm],
           xticks = [0,2,4,6,8,10,12,14,16,18,20],
           xlabel = 'time (minutes)',
           ylabel = 'conductivity (microS/cm)')

# "Switch" for selecting time based on arg
def analyte_time_set(ion):
    switcher = {
        "F" : 4.13 ,
        "Cl": 6.12,
        "Nitrite": 7.35,
        "Br": 9.38,
        "Nitrate": 10.81,
        "Phosphate": 13.42,
        "Sulfate": 15.63
    }
    return switcher.get(ion)

#Tolerance is because peaks seem to shift a bit between runs, which makes some get skipped; play with this if you're concerned about peak overlap
def peakfinder(snippet,analyte,tolerance = 0.3):
    t_expected = analyte_time_set(analyte)
    if analyte == "Nitrite":
        thresh = 0.8
        y = snippet[(snippet.time > t_expected - thresh) & (snippet.time < t_expected + thresh)].conductivity
    else: y = snippet.conductivity        

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
    
    ion_peak = []
    for peak in ind_peaks:
        p = snippet.iloc[peak[0]:peak[1]]
        pt = p.time.values.tolist()
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
def process_chroma_std(analyte, dimension, chatty = True):
    standard_data = []
    files = get_runfiles(True)
    
    print("Processing " + temp_fs + " standards for " + analyte)
    for fn in files:
        rt,sn,fr = process(fn)
        std_conc = int(sn.split()[1])
        std_analyte = sn.split()[3]
        max_cond = max(fr.conductivity)
        rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime
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
        else:
            if chatty == True: print(sn + ": No peaks")
            datarow = datarow + [np.nan,np.nan,np.nan]
        standard_data = standard_data + [datarow]

    # Create dataframe for analyte standards
    df_standards = pd.DataFrame(standard_data)
    df_standards.columns = ['analyte','conc_uM','analysis_time','max_cond','peak_area','peak_height','fwhm']
    df_standards.analysis_time=pd.to_datetime(df_standards.analysis_time) #Includes 100um Single-anion stds (not needed for calibration but good to have as evidence of peak timing)
    df_standards = df_standards[df_standards['analyte'] == 'mixedanion'].sort_values(["conc_uM","analysis_time"])
    
    st_square = df_standards.pivot(columns = "conc_uM",values = dimension).apply(lambda x: pd.Series(x.dropna().values))
    st_square = st_square[(np.abs(stats.zscore(st_square)) < 3).all(axis=1)]
    st_melt = st_square.melt(value_name = dimension)
    return st_melt

#Sitename options are presently ["GMP","CBG"], must match the folder names in the IC Data directory
def process_chromatogram(analyte,fn, chatty = True):
    rt,sn,fr = process(fn)  # Get data frame
    s_id = sn.split()[0]    # Get identifying metadata: sample ID, runtime, dilution
    max_cond = max(fr.conductivity)     # Identify the max conductivity to decide whether or not to analyze peaks (useful when there is instrument error)
    if temp_fs == "GMP":
        s_date = sn.split()[1]
        dil = int(sn.split()[2].strip('x'))
    if temp_fs == "CBG":
        dil = 1.0
    rt_l = datetime.strftime(rt,'%Y-%m-%d %H:%M:%S') #Long-form runtime

    if temp_fs == "GMP": datarow = [s_id,s_date,rt_l,dil,max_cond]  # Include metadata in data row
    else: datarow = [s_id,rt_l,dil,max_cond]
        
    if(max_cond < 100):                          # If max conductivity reasonable:
        cp, pb = peakfinder(fr,analyte)       # Find analyte peak
        if(len(cp) > 0):                         # If analyte peak exists
            pa, pm, fwhm = calc_peakstats(fr,cp,pb)           # Calculate area, height at max, full width at half max
            if chatty == True: print(sn + ": " + analyte + " peak area: " + str(pa))       # Status report
            datarow = datarow + [pa,pm,fwhm]            # Add data row to data file
        else:                                       # If no analyte peak:
            if chatty == True: print(sn + ": No " + analyte + " peak")     # status report
            datarow = datarow + [0,0,0]                 # All zeroes (no detectable analyte, but data is good)
    else:                                       # If max conductivity is excessive:
        if chatty == True: print(sn + ": No peaks")                    # Status report
        datarow = datarow + [np.nan,np.nan,np.nan]  # All not-a-number (data is bad)
    return datarow

# dimension options:
# "peak_area" or "peak_height"
def full_run(selected_analytes=potential_analytes, dimension = "peak_area", verbose=True):
    for temp_analyte in selected_analytes:

        st_reduced = process_chroma_std(temp_analyte,dimension,verbose)
        
        #pd_ststats = pd.DataFrame([st_square.mean(),st_square.median(),st_square.std(),st_square.min(),st_square.max()])
        #fit_resid = fit[1][0]
        fit = np.polyfit(st_reduced.conc_uM,st_reduced[dimension],1,full=True)       # Regress standard conc w/ chromatogram peak area
        plot_std(fit,st_reduced,temp_analyte,dimension)
        
        runfiles = get_runfiles()
        analysis_data = []
        
        print("Processing " + temp_fs + " samples for " + temp_analyte)
        for fn in runfiles:
            dr = process_chromatogram(temp_analyte,fn,chatty = verbose)
            analysis_data = analysis_data + [dr]
        df_d = pd.DataFrame(analysis_data)  
        
        if temp_fs == "GMP": df_d.columns = ['name','sample_date','analysis_time','dilution','max_cond','peak_area','peak_height','fwhm']
        else: df_d.columns = ['name','analysis_time','dilution','max_cond','peak_area','peak_height','fwhm']
                
        #formulae used:
        #dimension = conc * fit[0] + fit[1]
        #conc = (dimension - fit[1])/fit[0]
        df_d['conc_uM'] = ((df_d[dimension] - fit[0][1])/fit[0][0])*df_d['dilution']    # Use regression to calculate concentration from sample chromatogram peak area
        df_d.analysis_time=pd.to_datetime(df_d.analysis_time)
        df_d.sort_values("analysis_time",inplace = True)
        
        #Save data
        df_d.to_csv(processed_loc + "datafiles//" + "IC_2019_" + temp_analyte + "_"+ dimension + "_sampledata.csv",index = False)
        st_reduced.to_csv(processed_loc + "datafiles//" + "IC_2019_" + temp_analyte + "_"+ dimension + "_standards.csv",index = False)

def plot_std(fit_in,st_red,an,dim):
    #Plot standards with linear fit shown
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lin_x_l = 0
    lin_x_r = 100
    lin_y_l = lin_x_l * fit_in[0][0] + fit_in[0][1]
    lin_y_r = lin_x_r * fit_in[0][0] + fit_in[0][1]
    s_title = "Standard curve for "+ an + " using " + dim
    figname = an + "-" + dim +"_std_all.png"
    
    ax.plot(st_red.conc_uM, st_red[dim],'.')
    ax.plot([lin_x_l, lin_x_r], [lin_y_l, lin_y_r])
    ax.set(title = s_title,
           ylabel = dim,
           xlabel = 'concentration (uM)',
           ylim = [0,lin_y_r*1.2])
    plt.savefig(processed_loc + "figures//" +figname)
    plt.close()

def plot_all(p_stds = True):
    runfiles = get_runfiles(p_stds)
    for fn in runfiles:
        rt,sn,fr = process(fn)  # Get data frame
        rt_fn = datetime.strftime(rt, "%Y%m%d-%H%M%S") #No punctuation runtime (for filenames)
        plot_chroma(rt_fn,sn,fr)  
        snp = "_".join(sn.split())
        snp = "-".join(snp.split("/"))
        figname = rt_fn + "_" + snp +".png"
        plt.savefig(processed_loc + "figures//" + figname)
        plt.close()