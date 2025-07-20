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


# --- Directories ---
# # tintin
# CERRA_dir = '/work_big/users/clima/portal/CERRA-Land/'
# ERA5_dir = '/work_big/users/clima/portal/ERA5/'
# CRCM5_dir = '/work_big/users/clima/portal/CRCM5-LE/'
# fig_dir = '/home/portal/figures/case-studies_byNode/'

# alice
CERRA_dir = '/media/alice/Extreme SSD1/folders/data_CNR/CERRA-Land/'
ERA5_dir = '/media/alice/Extreme SSD1/folders/data_CNR/ERA5/'
CRCM5_dir = '/media/alice/Extreme SSD1/folders/data_CNR/CRCM5-LE/'
fig_dir = './figures/'


# --- Event and LE analogue definition ---
# Event
lselect = 'alert-regions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 1
no_event = 1
# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)

# Variable
varname = 'psl'

# Quantile and analogue spacing
qtl_LE = 0.99

# Number of ensemble members
no_membs = 2 

# Epochs
list_year_ranges = [[1955, 1974], [2004, 2023], [2030, 2049], [2080, 2099]] # past [1955-1974], present [2004-2023], near-future [2030-2049], far future [2080-2099]
no_epochs = len(list_year_ranges)

# Difference between epochs
diff_indices = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]  # Define the indices of epochs to compare

# List of members
list_membs = [name for name in os.listdir(CRCM5_dir + varname) if os.path.isdir(os.path.join(CRCM5_dir + varname, name))]
list_membs = sorted(list_membs)[:no_membs]  # Select the first 'no_membs' members


# --- Upload analogue dates and distances for each year range---
# Create a list of no_epochs dictionaries with member names as keys and their corresponding times and distances as values
ensemble_data = []
for i, year_range in enumerate(list_year_ranges):
    epoch_data = {}
    for memb in list_membs:
        # Construct the file path
        file_path = f'./analogue_data/times_distances_analogues-{varname}_node{no_node}-extreme{no_event}-alertregions_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5-LE_memb-{memb}.npz'
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
list_ds_clim = [] # Initialize an empty list to store climatology file paths
analogue_numbers = np.arange(1, no_analogues_LE + 1)  # Create an array of analogue numbers from 1 to n_analogues_LE
for i, year_range in enumerate(list_year_ranges):

    # List file paths for current epoch
    anom_files_epoch, clim_files_epoch = fanPM.get_anomaly_climatology_paths_CRCM5(CRCM5_dir, varname, list_membs, year_range)
    
    # Lists of datasets for current epoch
    anom_sel = fanPM.open_member_datasets(anom_files_epoch, combine='by_coords', expand_member_dim=True)
    clim_sel = fanPM.open_member_datasets(clim_files_epoch, combine='by_coords', expand_member_dim=True)
    
    # Select analogue days for current epoch by member
    anom_analogues = []
    clim_analogues = []
    for im, memb in enumerate(list_membs):
        # Select the times and doys of the analogues for the current member
        times_analogues = ensemble_data[i][memb]['times']
        doys_analogues = np.vectorize(lambda d: d.timetuple().tm_yday)(times_analogues)
        # Extract doys from climatology dataset
        doys_clim = np.vectorize(lambda d: d.timetuple().tm_yday)(clim_sel[im].time.values)
        # Select the anomaly and climatology datasets for the current member
        anom_memb = anom_sel[im].sel(time=anom_sel[im].time.isin(times_analogues)).assign_coords(time=analogue_numbers)
        anom_analogues.append(anom_memb)
        clim_memb = [clim_sel[im].sel(time=clim_sel[im].time[doys_clim == doy]) for doy in doys_analogues]
        clim_memb = xr.concat(clim_memb, dim="time").assign_coords(time=analogue_numbers)
        clim_analogues.append(clim_memb)
    
    # Concatenate the anomaly and climatology datasets for the current epoch
    ds_anom_analogues = xr.concat(anom_analogues, dim='member')[varname] * 0.01
    ds_anom_analogues = ds_anom_analogues.rename({"time": "analogue"})
    ds_clim_analogues = xr.concat(clim_analogues, dim='member')[varname] * 0.01
    ds_clim_analogues = ds_clim_analogues.rename({"time": "analogue"})
    
    # Select event box
    lon_mask, lat_mask = fanPM.lonlat_mask(ds_anom_analogues.lon.values, ds_anom_analogues.lat.values, box_event)
    mask = lat_mask[:, np.newaxis] & lon_mask
    mask_xr = xr.DataArray(
        mask,
        dims=["lat", "lon"],
        coords={"lat": ds_anom_analogues.lat.values, "lon": ds_anom_analogues.lon.values},
    )
    ds_anom_analogues = ds_anom_analogues.where(mask_xr, drop=True)
    ds_clim_analogues = ds_clim_analogues.where(mask_xr, drop=True)

    # Save in list by epoch
    list_ds_anom.append(ds_anom_analogues)
    list_ds_clim.append(ds_clim_analogues)
        
# # Print dataset for the first epoch
# print(f"Anomaly dataset for epoch {list_year_ranges[0]}: {list_ds_anom[0]}")
# print(f"Climatology dataset for epoch {list_year_ranges[0]}: {list_ds_clim[0]}")


# --- Compute LE analogue differences ---
# Compute differences between LE analogues from different epochs
list_ds_diff = []  # Initialize an empty list to store the differences
for i, (epoch1, epoch2) in enumerate(diff_indices):
    # Compute the difference between the two epochs
    ds_epoch1 = (list_ds_clim[epoch1] + list_ds_anom[epoch1]).mean(dim=('member','analogue'))
    ds_epoch2 = (list_ds_clim[epoch2] + list_ds_anom[epoch2]).mean(dim=('member','analogue'))
    ds_diff = ds_epoch2 - ds_epoch1
    # Add epoch information to the dataset
    ds_diff.attrs['epoch1'] = f"{list_year_ranges[epoch1][0]}-{list_year_ranges[epoch1][1]}"
    ds_diff.attrs['epoch2'] = f"{list_year_ranges[epoch2][0]}-{list_year_ranges[epoch2][1]}"
    list_ds_diff.append(ds_diff)


# --- Save differences to NetCDF files ---
output_dir = './analogue_data/analogue_differences/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 
# Save each difference dataset to a NetCDF file
for i, ds_diff in enumerate(list_ds_diff):  
    output_file = f"{output_dir}LE-analogue-difference_{varname}_node{no_node}-extreme{no_event}-alertregions_{int(qtl_LE*100)}pct_diff{ds_diff.attrs['epoch2']}_{ds_diff.attrs['epoch1']}_CRCM5_{no_membs}membs.nc"
    ds_diff.to_netcdf(output_file)
