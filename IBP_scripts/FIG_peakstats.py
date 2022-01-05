# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:35:29 2021

@author: Packman-Field
"""



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import seaborn as sns

pd.set_option('display.max_columns', 500)
sns.color_palette("rocket")
sns.set_style("ticks")
sns.set(font_scale = 2)

paperII_path = "C:\Users\Packman-Field\Documents\Paper II\Water Data\\"

s = ["WLW1","WLW2","WLW3","WLW4","WLW5","WLW6","WLW7","WLW8","WLW9","WLW10","WLW12","WLW13"] #Figure out what's up with 14
s_group = ["upland","wetland","wetland","wetland","wetland","wetland","ridge","ridge","ridge","upland","wetland","upland"]
df_locs = pd.DataFrame({"sensor_id":s,"group":s_group})
# 1-upland, 2-wetland/swale, 3-sandridge

pt = 0.002

peakstats = pd.read_csv(paperII_path + "peakstats_thresh " + str(pt) + "m.csv", index_col = 0, parse_dates = ['x_0_dt'])

ps_simple = peakstats[["sensor_id","x_0_dt",'x_day']].copy()

ps_simple['duration_hr'] = (peakstats.x_max-peakstats.x_0)/3600
ps_simple['ttp_hr'] = (peakstats.x_peak - peakstats.x_0)/3600     #time to peak
ps_simple['rx_hr'] = (peakstats.x_max - peakstats.x_peak)/3600     #relax

ps_simple['elev_init_m'] = peakstats.y_sensor + peakstats.y_0
ps_simple['rise_peak_m'] = peakstats.y_peak - peakstats.y_0

ps_simple = ps_simple.merge(df_locs,on ="sensor_id")

#sns.lmplot(x = "x_day", y = "elev_peak_m", data = ps_simple, hue = "sensor_id")

smp = pd.read_csv(paperII_path + "SMP2_ibp_main.csv",parse_dates = ["date_time"])
smp["top50"] = smp[["a1_moisture","a2_moisture","a3_moisture"]].mean(axis = 1)
ps_antecedent = ps_simple.merge(smp[["date_time","top50"]] ,left_on = "x_0_dt", right_on = "date_time").dropna(axis = 0)


precip = pd.read_csv(paperII_path + 'precip_in_Crete.csv', parse_dates = ['date_time'])
precip['precip_cm'] = precip['precip_in']*2.54
precip = precip[['date_time','precip_cm']].copy().set_index('date_time').resample("30T").sum().reset_index()
precip['P24hr_cm'] = precip['precip_cm'].rolling(48).sum().round(6)
precip['P48hr_cm'] = precip['precip_cm'].rolling(96).sum().round(6)
ps_antecedent = ps_antecedent.merge(precip[["date_time","P24hr_cm","P48hr_cm"]],left_on = "x_0_dt", right_on = "date_time").dropna(axis = 0)

ps_antecedent = ps_antecedent[ps_antecedent.P24hr_cm > 0]
#ps_antecedent.drop(["date_time_x","date_time_y"],axis = 1)

ps_choice = ps_antecedent[["ttp_hr","rise_peak_m","elev_init_m","top50","P24hr_cm","group","sensor_id","x_day"]]
#sns.pairplot(ps_choice, 
#             hue = "sensor_id", plot_kws={'alpha': 0.75}, hue_order = s, palette = "rocket")

sns.pairplot(ps_antecedent[["elev_init_m","rise_peak_m","sensor_id","x_day"]], 
             hue = "sensor_id", hue_order = s, palette = "rocket")
sns.scatterplot(x = "elev_init_m", y = "rise_peak_m",data = ps_choice, 
             s = 200, hue = "sensor_id", hue_order = s, palette = "rocket", alpha = 0.8)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)

sns.scatterplot(x = "top50", y = "ttp_hr",data = ps_choice, 
             s = 200, hue = "group", hue_order = ["wetland","ridge","upland"], alpha = 0.8)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)

plt.Figure()
sns.scatterplot(x = "P24hr_cm", y = "rise_peak_m",data = ps_choice, 
             s = 200, hue = "group", hue_order = ["wetland","ridge","upland"], alpha = 0.8)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)


sns.scatterplot(x = "x_day", y = "elev_init_m",data = ps_choice, 
             s = 200, hue = "group",hue_order = ["wetland","ridge","upland"], alpha = 0.8)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.)



