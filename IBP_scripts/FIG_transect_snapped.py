# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:15:28 2019

@author: Packman-Field
"""

import pandas as pd
import matplotlib.pyplot as plt

spatial_datapath = 'C:\Users\Packman-Field\Desktop\Transect_Data\\'

ts_ground = pd.read_csv(spatial_datapath + "ts_1m_elev.csv")
ts_ground.drop("OBJECTID", axis = 1, inplace = True)
ts_ground.columns = ["x_dist_m","latitude","longitude","z_ft"]
ts_ground["z_m"] = ts_ground["z_ft"]*0.3048

ts_wells = pd.read_csv(spatial_datapath + "WL_trans_dist.csv")
ts_wells.drop(["OBJECTID","NEAR_FID","NEAR_X","NEAR_Y","date","type"], axis = 1, inplace = True)
ts_wells.drop_duplicates("sensor",inplace = True)
ts_wells = ts_wells[~ts_wells.sensor.str.contains("S")]
ts_wells.rename({"NEAR_DIST":"x_dist_m"},axis = 1,inplace = True)
ts_wells["z_sensor_m"] = (ts_wells["top_elev_ft"]-ts_wells["cable_len_ft"])*0.3048

ts_wells2 = pd.read_csv(spatial_datapath + "WaterLevel_TransectPoints.csv")
ts_wells2.drop(["OBJECTID","NEAR_FID","NEAR_X","NEAR_Y","date","type"], axis = 1, inplace = True)
ts_wells2.drop_duplicates("sensor",inplace = True)
ts_wells2 = ts_wells2[~ts_wells2.sensor.str.contains("S")]

ts_wells = ts_wells.merge(ts_wells2, on = 'sensor')
ts_wells.drop(ts_wells[ts_wells.NEAR_DIST > 200].index)

plt.plot(ts_ground.x_dist_m,ts_ground.z_m)
plt.plot(ts_wells["x_dist_m"],ts_wells["z_sensor_m"],'.')