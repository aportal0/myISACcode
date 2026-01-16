# run .py file from conda environment xesmf_env

# --- Imports ---
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
from dask.distributed import Client, LocalCluster
from scipy.interpolate import griddata
import calendar
from datetime import datetime, timedelta
import cftime

# --- Custom Functions ---
# sys.path.append('/home/portal/script/python/precip_Cristina/')                    # tintin
sys.path.append('/home/alice/Desktop/work/git/myISACcode/python/precip_Cristina')   # alice
import functions_analogues_PrMax as fanPM
import functions_analogues_LUCAFAMOSS as fan


# --- Directories ---

# # tintin
# CERRA_dir = '/work_big/users/clima/portal/CERRA-Land/'
# ERA5_dir = '/work_big/users/clima/portal/ERA5/'
# CRCM5_dir = '/work_big/users/clima/portal/CRCM5-LE/'

# alice
CERRA_dir = "/media/alice/Crucial X9/portal/data_CNR/CERRA/"
ERA5_dir = "/media/alice/Crucial X9/portal/data_CNR/ERA5/"
CRCM5_dir = "/media/alice/Crucial X9/portal/data_CNR/CRCM5-LE/"
analogue_dir = "/home/alice/Desktop/work/git/myISACcode/python/analogues/analogue_data/analogue_times_distances/"


# --- Parameters LE analogue search ---

# Variable
var_analogues = ['mslp'] # ['mslp','z500']
var_names = ['msl'] # ['msl','z']
var_factors = [0.01]  # [0.01, 1/9.81] to convert from Pa to hPa and from geopot to geopot height
str_vars = '-'.join(var_analogues)+'-std'

# Quantile and analogue spacing
qtl = 0.99
analogue_spacing = 7 # days

# Time
year_range = [2004, 2023]
years_sel = np.arange(year_range[0], year_range[1]+1)


# --- Event Definition ---

# Event
lselect = 'alertregions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 5
no_event = 4
str_event = f'node{no_node}-extreme{no_event}-{lselect}'
# Upload ERA5 info
df_events = pd.read_excel(CERRA_dir+'events_cum_on_above99_alertregions_CERRA.xlsx', sheet_name=no_node-1)
time_event = df_events['Time'].iloc[no_event-1] + pd.Timedelta('12h')
doy_event =  time_event.timetuple().tm_yday

# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)

# Months for analogue selection
month_event = time_event.month
months_sel = [month_event-1, month_event, month_event+1]
month_names = [calendar.month_abbr[month] for month in months_sel]
str_months = ''.join([name[0] for name in month_names])


# --- Load event data ---

# From ERA5
dmslp_event = fanPM.load_ERA5_data_regridded_to_CRCM5('mslp', 'daily', time_event, box_event, l_anom=True, l_detrend=False,data_dir=ERA5_dir+'mslp/res05/')
mslp_event_clim = fanPM.load_ERA5_clim_regridded_to_CRCM5('mslp', doy_event, box_event, l_smoothing=True, data_dir=ERA5_dir+'mslp/res05/climatology/')
dz500_event = fanPM.load_ERA5_data_regridded_to_CRCM5('z500', 'daily', time_event, box_event, l_anom=True, l_detrend=True, data_dir=ERA5_dir+'z500/res05/')
z500_event_clim = fanPM.load_ERA5_clim_regridded_to_CRCM5('z500', doy_event, box_event, l_smoothing=True, data_dir=ERA5_dir+'z500/res05/climatology/')
print("Event data loaded...")


# --- Load ERA5 data and compute Euclidean distance from event ---

# Fix number of analogues per member based on the quantile
n_analogues = 18 #int(np.round(len(dmslp_sel.time) * (1-qtl)))
# Load ERA5 data and select time range and event box and standardise
# Compute Euclidean distance from event to all selected data
dist_per_var = {}
dist = None
for iv, var in enumerate(var_analogues):
    if var=='z500':
        file_pattern = ERA5_dir + var +"/res05/ERA5_"+var+"_NH_daily_*_anom_detrended_regridded-to-CRCM5.nc"
    else:
        file_pattern = ERA5_dir + var +"/res05/ERA5_"+var+"_NH_daily_*_anom_regridded-to-CRCM5.nc"
    anom_tmp = xr.open_mfdataset(file_pattern, combine='by_coords', parallel=True)[var_names[iv]] * var_factors[iv]
    
    # Select time range
    anom_tmp = anom_tmp.sel(time=anom_tmp.time.dt.month.isin(months_sel))
    anom_tmp = anom_tmp.sel(time=anom_tmp.time.dt.year.isin(years_sel))

    # Select event box
    lon_mask, lat_mask = fanPM.lonlat_mask(anom_tmp.lon.values, anom_tmp.lat.values, box_event)
    mask = lat_mask[:, np.newaxis] & lon_mask
    mask_xr = xr.DataArray(
        mask,
        dims=["lat", "lon"],
        coords={"lat": anom_tmp.lat, "lon": anom_tmp.lon},
    )
    anom_tmp = anom_tmp.where(mask_xr)
    vars()[f'd{var}_sel'] = anom_tmp.dropna(dim="lat", how="all").dropna(dim="lon", how="all")

    # Standardise event and selected data
    stdev = vars()[f'd{var}_sel'].std(dim='time').squeeze()
    anom_std = (vars()[f'd{var}_sel'] / stdev).squeeze()
    anom_event_std = (vars()[f'd{var}_event'] / stdev).squeeze()

    # Compute euclidean distance from the event to the selected data
    dist_per_var[var] = fan.function_distance(anom_event_std, anom_std, nan_version=True)
    if dist is None:
        dist = dist_per_var[var]**2
    else:
        dist += dist_per_var[var]**2
    print(f"Distance computed for variable {var}...")
dist = np.sqrt(dist)
print("Distance computation completed...")

# --- Search for analogues ---

# All times and event time
all_times = vars()[f'd{var_analogues[0]}_sel'].time.values
event_time = vars()[f'd{var_analogues[0]}_event'].time.values

# First search of n_analogues
factor_0sel = 2
qtl_0sel = 1 - ((1 - qtl) * factor_0sel)  # First selection of the quantile, for extracting a total of n_analogues
l_0sel = True  # Flag for selection of analogues

while l_0sel:
    # Compute log-transformed distance
    logdist = np.log(1 / dist)
    # Threshold at given quantile 0sel
    thresh_0sel = np.percentile(logdist, qtl_0sel * 100, axis=0)
    mask_analogues = logdist >= thresh_0sel
    
    # Exclude analogue times within ±7 days of their associated event
    time_diff = np.abs((all_times - event_time).astype('timedelta64[D]').astype(int))
    mask_analogues &= np.array(time_diff) >= analogue_spacing  # update mask to exclude times too close to the event
    indices_analogues = np.where(mask_analogues)[0]  # indices of all valid analogue times
    
    # Filter analogues based on the mask and logdist, ensuring they are spaced correctly (analogue_spacing days apart)
    indices_filtered_analogues = fan.timefilter_analogues(indices_analogues, logdist, all_times, analogue_spacing)

    if len(indices_filtered_analogues)>=n_analogues:
        indices_filtered_analogues = indices_filtered_analogues[:n_analogues]  # Select the first n_analogues (corresponding to the quantile qtl)
        l_0sel = False
        print("Selection completed using pool data from quantile", qtl_0sel)
    else:
        factor_0sel +=1
        qtl_0sel = 1 - ((1 - qtl) * factor_0sel)

# Save distance and times of selected analogues
dist_filtered_analogues = dist[indices_filtered_analogues]
times_filtered_analogues = all_times[indices_filtered_analogues]


# --- Save analogue data ---

# Create file path
npz_file_path = f'{analogue_dir}times_distances_analogues-{str_vars}_{str_event}_{int(qtl*100)}pct_{year_range[0]}-{year_range[1]}_ERA5.npz'
np.savez(npz_file_path, 
         times=times_filtered_analogues, 
         distances=dist_filtered_analogues)