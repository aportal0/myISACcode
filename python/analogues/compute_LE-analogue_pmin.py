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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

# --- Custom Functions ---
# sys.path.append('/home/portal/script/python/precip_Cristina/')                    # tintin
sys.path.append('/home/alice/Desktop/work/git/myISACcode/python/precip_Cristina')   # alice
sys.path.append('/home/alice/Desktop/work/git/myISACcode/python')                   # alice
import functions_analogues_PrMax as fanPM
import functions_analogues_LUCAFAMOSS as fan

# --- Warning settings ---
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Directories ---
# # tintin
# CERRA_dir = '/work_big/users/clima/portal/CERRA-Land/'
# ERA5_dir = '/work_big/users/clima/portal/ERA5/'
# CRCM5_dir = '/work_big/users/clima/portal/CRCM5-LE/'
# fig_dir = '/home/portal/figures/analogues/'

# alice
CERRA_dir = '/media/alice/Crucial X9/portal/data_CNR/CERRA-Land/'
ERA5_dir = '/media/alice/Crucial X9/portal/data_CNR/ERA5/'
CRCM5_dir = '/media/alice/Crucial X9/portal/data_CNR/CRCM5-LE/'
fig_dir = '/home/alice/Desktop/CNR/ENCIRCLE/materiale_alice/figures/analogues/'
output_dir = './analogue_data/analogue_differences/'  # Directory to save the output files
output_pslmin_dir = './analogue_data/psl_min/'  # Directory to save the regional mean files

if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if not os.path.exists(output_pslmin_dir):
    os.makedirs(output_pslmin_dir)

# --- Event and LE analogue definition ---
# Event
lselect = 'alertregions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 3
no_event = 3
event_origin = 'ERA5'  # 'ERA5' or 'CRCM5-LE'
if event_origin == 'ERA5':
    str_event = f'node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'
elif event_origin == 'CRCM5-LE':
    str_event = f'BAM-node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'
# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)

# Variable
varname = 'psl' # Variable to compute the difference between analogues, e.g. 'zg' for geopotential height
var_analogues = 'psl'  # Variable used to find the analogues, e.g. 'psl' for sea level pressure
var_factor = 0.01  # Factor to convert the variable to the correct units (e.g., psl from Pa to hPa)

# Quantile and analogue spacing
qtl_LE = 0.99

# Number of ensemble members
no_membs = 49

# Epochs
list_year_ranges = [[1955, 1974], [2004, 2023], [2080, 2099]] # past [1955-1974], present [2004-2023], near-future [2030-2049], far future [2080-2099]
no_epochs = len(list_year_ranges)

# Difference between epochs
diff_indices = [[0,1],[0,2],[1,2]]  # Define the indices of epochs to compare

# List of members
list_membs = [name for name in os.listdir(CRCM5_dir + 'psl') if os.path.isdir(os.path.join(CRCM5_dir + 'psl', name))]
list_membs = sorted(list_membs)[:no_membs]  # Select the first 'no_membs' members


# --- Upload analogue dates and distances for each year range---
# Create a list of no_epochs dictionaries with member names as keys and their corresponding times and distances as values
ensemble_data = []
for i, year_range in enumerate(list_year_ranges):
    epoch_data = {}
    for memb in list_membs:
        # Construct the file path
        file_path = f'./analogue_data/times_distances_analogues-{var_analogues}_{str_event}_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5-LE_memb-{memb}.npz'
        # Load the data from the npz file
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        times = data['times']
        distances = data['distances']
        epoch_data[memb] = {'times': times, 'distances': distances}
    ensemble_data.append(epoch_data)
no_analogues_LE = len(ensemble_data[0][list_membs[0]]['times'])  # Number of analogues per member
# # Print the ensemble data for first epoch
# print(f"Ensemble data for epoch {list_year_ranges[0]}: {ensemble_data[0]}")


# --- Load LE analogue data for all epochs ---
# Anomalies and climatology
# List file paths
list_ds_anom = []  # Initialize an empty list to store file paths
analogue_numbers = np.arange(1, no_analogues_LE + 1)  # Create an array of analogue numbers from 1 to n_analogues_LE
for i, year_range in enumerate(list_year_ranges):

    # List file paths for current epoch
    list_times = [data['times'] for data in ensemble_data[i].values()]
    anom_files_epoch, clim_files_epoch = fanPM.get_anomaly_climatology_paths_CRCM5_bymonth(CRCM5_dir, varname, list_membs, list_times)

    # Lists of datasets for current epoch
    anom_sel = fanPM.open_member_datasets(anom_files_epoch, combine='by_coords', expand_member_dim=True)

    # Select analogue days for current epoch by member
    anom_analogues = []
    for im, memb in enumerate(list_membs):
        # Select the times and doys of the analogues for the current member
        times_analogues = ensemble_data[i][memb]['times']
        # Select the anomaly and dataset for the current member
        anom_memb = anom_sel[im].sel(time=times_analogues)
        no_analogues = len(anom_memb.time)  # Number of analogues for the current member
        analogue_numbers = np.arange(1, no_analogues+1)  # Create an array of analogue numbers for the current member
        anom_memb = anom_memb.assign_coords(time=analogue_numbers).rename({'time': 'analogue'}).chunk({'analogue': -1, 'lat': 200, 'lon': 200})
        # Compute pmin in box (excluding NaNs)
        min_anom = anom_memb.min(dim=['lat', 'lon'], skipna=True)
        min_anom['psl'] = min_anom['psl'] * var_factor
        # Compute min value from mean psl in box
        boxmean = anom_memb.mean(dim=['lat', 'lon'], skipna=True)
        min_from_boxmean = (anom_memb - boxmean).min(dim=['lat', 'lon'], skipna=True)
        min_from_boxmean['psl'] = min_from_boxmean['psl'] * var_factor

        # Save regional m to NetCDF file
        suffix_file = f"{varname}_{str_event}_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5_{memb}.nc"
        min_file = f"{output_pslmin_dir}analogues-{var_analogues}_min-{suffix_file}"
        min_from_mean_file = f"{output_pslmin_dir}analogues-{var_analogues}_min-from-boxmean-{suffix_file}"
        if os.path.exists(min_file):
            print(f"Pmin of epoch {i} already exists: {min_file}")
        else:
            min_anom.to_netcdf(min_file)
            print(f"Saved Pmin of epoch {i}, member {memb}")
        if os.path.exists(min_from_mean_file):
            print(f"Pmin from box mean of epoch {i} already exists: {min_from_mean_file}")
        else:
            min_from_boxmean.to_netcdf(min_from_mean_file)
            print(f"Saved Pmin from box mean of epoch {i}, member {memb}")
