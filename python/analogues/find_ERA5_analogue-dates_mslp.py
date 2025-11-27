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
analogue_dir = "/home/alice/Desktop/work/git/myISACcode/python/analogues/analogue_data/"


# --- Parameters LE analogue search ---

# Variable
varname = 'mslp'

# Quantile and analogue spacing
qtl = 0.99
analogue_spacing = 7 # days

# Time
year_range = [2004, 2023]
years_sel = np.arange(year_range[0], year_range[1]+1)


# --- Event Definition ---

# Event
lselect = 'alertregions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 3
no_event = 3
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
dmslp_event = fanPM.load_ERA5_data('mslp', 'daily', time_event, box_event, l_anom=True, data_dir=ERA5_dir+'mslp/')
mslp_clim = fanPM.load_ERA5_clim('mslp', doy_event, box_event, l_smoothing=True, data_dir=ERA5_dir+'mslp/climatology/')
# Regrid the data to the desired resolution
dmslp_event = fanPM.regrid_with_xesmf(dmslp_event, box_event, resolution=0.5)
mslp_event_clim = fanPM.regrid_with_xesmf(mslp_clim, box_event, resolution=0.5)
print("Event data loaded...")


# --- Load ERA5 data and compute Euclidean distance from event ---

# Define the file pattern for loading mslp data
file_pattern = ERA5_dir + varname +"/res05/ERA5_"+varname+"_NH_daily_*_anom_regridded-to-CRCM5.nc"
dmslp_tmp = xr.open_mfdataset(file_pattern, combine='by_coords', parallel=True)['msl'] * 0.01

# Select time range
dmslp_tmp = dmslp_tmp.sel(time=dmslp_tmp.time.dt.month.isin(months_sel))
dmslp_tmp = dmslp_tmp.sel(time=dmslp_tmp.time.dt.year.isin(years_sel))

# Select event box
lon_mask, lat_mask = fanPM.lonlat_mask(dmslp_tmp.lon.values, dmslp_tmp.lat.values, box_event)
mask = lat_mask[:, np.newaxis] & lon_mask
mask_xr = xr.DataArray(
    mask,
    dims=["lat", "lon"],
    coords={"lat": dmslp_tmp.lat, "lon": dmslp_tmp.lon},
)
dmslp_sel = dmslp_tmp.where(mask_xr, drop=True)

# Compute euclidean distance from the event to the selected mslp data
dist = fan.function_distance(dmslp_event, dmslp_sel, nan_version=True)

# Fix number of analogues per member based on the quantile
n_analogues = int(np.round(len(dmslp_sel.time) * (1-qtl)))


# --- Search for analogues ---

# All times and event time
all_times = dmslp_sel.time.values
event_time = dmslp_event.time.values

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
npz_file_path = f'{analogue_dir}times_distances_analogues-{varname}_{str_event}_{int(qtl*100)}pct_{year_range[0]}-{year_range[1]}_ERA5.npz'
np.savez(npz_file_path, 
         times=times_filtered_analogues, 
         distances=dist_filtered_analogues)