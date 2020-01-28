import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os as os
import numpy as np

data_path = 'C:\Users\Vivien\Dropbox\IBP_Python\scripts_and_filenames\\'

pd.set_option('max_colwidth',100)
pd.options.display.max_rows = 20

sensor_meta = pd.read_table('sensor_elevations.csv',sep=',',index_col=False)
sensor_meta = pd.concat([sensor_meta,data_path+'processed_data\\'+sensor_meta.sensor+"_ibp_main.csv"],axis=1)
sensor_meta.columns=['data_id','top_elev_ft','cable_length_ft','ground_elev_ft','path']


def loop_clean(sensor_id = "all",f=None):
    if(sensor_id == "all"):
        print("Default settings, loop through all sensors.")
        #Iterate through all sensors and do things with them
        for index,row in sensor_meta.iterrows():
            print row['data_id']
            datafile = pd.read_csv(row['path'],index_col = 0)
            datafile.date_time = pd.to_datetime(datafile.date_time)
            datafile.loc[datafile.qual_c==0,"WS_elevation_m"]=np.nan #Skip qc-flagged values        
        
        print "Stitching PDF"
        name_string = data_path+"figures\\filename.pdf"
        multipage(name_string)
        
    elif(not sensor_meta.data_id.str.contains(sensor_id).any()):
        print("There is no data file for this sensor!")      
        #Stops here!
        
    else:
        print("Do action for sensor "+sensor_id+" only.")
        datafile = pd.read_csv(sensor_meta[sensor_meta.data_id == sensor_id].path.item(),index_col = 0)
        datafile.loc[datafile.qual_c==0,"WS_elevation_m"]=np.nan #Skip qc-flagged values
        
        #Do a function here which uses the datafile



 
#If there are plots, stitch them together as a PDF.       
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf',bbox_inches = 'tight',pad_inches = 0)
    pp.close()
    plt.close("all")
    os.startfile(filename)