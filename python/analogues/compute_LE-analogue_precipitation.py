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
from datetime import timedelta
import gc
import glob

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
# fig_dir = '/home/portal/figures/case-studies_byNode/'

# alice
CERRA_dir = '/media/alice/Crucial X9/portal/data_CNR/CERRA-Land/'
ERA5_dir = '/media/alice/Crucial X9/portal/data_CNR/ERA5/'
CRCM5_dir = '/media/alice/Crucial X9/portal/data_CNR/CRCM5-LE/'
fig_dir = '/home/alice/Desktop/CNR/ENCIRCLE/materiale_alice/figures/analogues/'
output_dir = './analogue_data/analogue_differences/'  # Directory to save the output files
output_mask_dir = './analogue_data/pr_in_mask/'  # Directory to save the regional mean files

if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

# --- Event and LE analogue definition ---
# Event
lselect = 'alertregions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 6
no_event = 19

event_origin = 'CRCM5-LE'  # 'ERA5' or 'CRCM5-LE'
if event_origin == 'ERA5':
    str_event = f'node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'
elif event_origin == 'CRCM5-LE':
    str_event = f'BAM-node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'

# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)

# Variable
varname = 'pr' # Variable to compute the difference between analogues, e.g. 'zg' for geopotential height
var_analogues = 'psl'  # Variable used to find the analogues, e.g. 'psl' for sea level pressure

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


# --- Load LE analogue data for all epochs and save mean per epoch ---
# Anomalies and climatology
# List file paths
list_ds_pr = []  # Initialize an empty list to store data sets for each epoch
analogue_numbers = np.arange(1, no_analogues_LE+1)  # Create an array of analogue numbers
# Precompute the mask only once (from first epoch)
mask_xr = None
mask_pr = None

for i, year_range in enumerate(list_year_ranges):

    # List file paths for current epoch
    list_times = [data['times'] for data in ensemble_data[i].values()]
    pr_files = fanPM.get_precipitation_paths_CRCM5_bymonth(CRCM5_dir, list_membs, list_times)

    # Lists of datasets for current epoch
    pr_sel = fanPM.open_member_datasets(pr_files, combine='by_coords', expand_member_dim=True)
    
    # Select analogue days for current epoch by member
    pr_analogues = []
    for im, memb in enumerate(list_membs):
        # Select the times and doys of the analogues for the current member
        times_analogues = ensemble_data[i][memb]['times'] + timedelta(hours=12)
        missing_times = [t for t in times_analogues if t not in pr_sel[im].time.values]
        if len(missing_times) > 0:
            print(f"Warning: {len(missing_times)} analogue times not found in pr_sel[{im}].time and will be removed:")
            for t in missing_times:
                print(f"  - {t}")
            # Remove missing times from times_analogues
            times_analogues = [t for t in times_analogues if t not in missing_times]
        # Select the anomaly and climatology datasets for the current member
        pr_memb = pr_sel[im].sel(time=times_analogues)
        # Assign analogue numbers to the time coordinate    
        no_analogues = len(pr_memb.time)  # Number of analogues for the current member
        analogue_numbers = np.arange(1, no_analogues+1)  # Create an array of analogue numbers for the current member
        pr_memb = pr_memb.assign_coords(time=analogue_numbers).rename({'time': 'analogue'})

        # Select event box
        if mask_xr is None:  # For the first epoch, select the box from the event
            lon_mask, lat_mask = fanPM.lonlat_mask(pr_memb.lon.values, pr_memb.lat.values, box_event)
            mask = lat_mask[:, np.newaxis] & lon_mask
            mask_xr = xr.DataArray(
                mask,
                dims=["lat", "lon"],
                coords={"lat": pr_memb.lat.values, "lon": pr_memb.lon.values},
            )
        pr_memb = pr_memb.where(mask_xr, drop=True).chunk({'analogue': -1, 'member': 1, 'lat': -1, 'lon': -1})
        
        # Append to list
        pr_analogues.append(pr_memb)

        # Compute and save regional mean precipitation in mask
        if mask_pr is None:
            if event_origin == 'ERA5':
                pr_mask = xr.open_dataset(f'./analogue_data/event_data/{varname}-mask_{str_event}_CERRA.nc')
            elif event_origin == 'CRCM5-LE':
                pattern = f'./analogue_data/BAM_data/{varname}-mask_BAM-{var_analogues}_{str_event}_*_2004-2023_CRCM5-LE_49membs.nc'
                mask_files = glob.glob(pattern)
                pr_mask = xr.open_dataset(mask_files[0])  # Assuming there's only one matching file   
            # weights = cos(lat)
            weights = np.cos(np.deg2rad(pr_mask['lat']))
            weights = weights.broadcast_like(pr_mask)   # expand to lat/lon grid
        pr_masked = pr_memb.where(pr_mask['pr_mask']==1)  # Apply mask to the precipitation data
        # now do a weighted mean over the region
        pr_memb_regional_mean = pr_masked.weighted(weights).mean(dim=("lat","lon")).squeeze()
        # Save regional mean to NetCDF file
        suffix_file = f"{varname}_{str_event}_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5_{memb}.nc"
        pr_regional_file = f"{output_mask_dir}analogues-{var_analogues}_mask-mean-{suffix_file}"
        if os.path.exists(pr_regional_file):
            print(f"Regional mean of epoch {i} already exists: {pr_regional_file}")
        else:
            pr_memb_regional_mean[varname].to_netcdf(pr_regional_file)
            print(f"Saved regional mean of epoch {i}, member {memb}")

    
    # Concatenate the datasets for the current epoch
    ds_pr_analogues = xr.concat(pr_analogues, dim='member')[varname].chunk({'analogue': -1, 'member': -1, 'lat': 5, 'lon': 5})
    # Save in list by epoch
    list_ds_pr.append(ds_pr_analogues) 

    # Save epoch mean to NetCDF file
    pr_epoch = ds_pr_analogues.mean(dim=('member','analogue'))
    suffix_file = f"{varname}_{str_event}_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5_{no_membs}membs.nc"
    epoch_file = f"{output_dir}analogues-{var_analogues}_{suffix_file}"
    if os.path.exists(epoch_file):
        print(f"Epoch file already exists: {epoch_file}")
    else:   
        pr_epoch.to_netcdf(epoch_file)
        print(f"Saved mean of epoch {i}")


# # --- Kolmogorov-Smirnov test for significance ---
# 
# # Perform the Kolmogorov-Smirnov test for each pair of epochs
# for i, (epoch1, epoch2) in enumerate(diff_indices):
#     print(f"Computing KS test for epoch {epoch1} vs epoch {epoch2}")
#     # Get the datasets for the two epochs
#     ds_epoch1 = list_ds_pr[epoch1].isel(member=slice(0, no_membs))
#     ds_epoch2 = list_ds_pr[epoch2].isel(member=slice(0, no_membs))
#     
#     ds1_flat = ds_epoch1.stack(analogue_all=('member', 'analogue')).chunk({'analogue_all': -1})
#     ds2_flat = ds_epoch2.stack(analogue_all=('member', 'analogue')).chunk({'analogue_all': -1})
#     
#     # Perform the Kolmogorov-Smirnov test for each grid point
#     ks_statistics = xr.apply_ufunc(
#         fanPM.ks_stat_and_pval,
#         ds1_flat,
#         ds2_flat,
#         input_core_dims=[['analogue_all'], ['analogue_all']],
#         output_core_dims=[['output']],
#         output_sizes={"output": 2},
#         vectorize=True,
#         dask='parallelized',
#         output_dtypes=[float],
#     )
#     ks_statistics = ks_statistics.assign_coords(output=["diff_statistic", "pvalue"])
# 
#     # Compute epoch differences
#     ds_diff = ds_epoch2.mean(dim=('member', 'analogue')) - ds_epoch1.mean(dim=('member', 'analogue'))
#     ds_diff.attrs['epoch1'] = f"{list_year_ranges[epoch1][0]}-{list_year_ranges[epoch1][1]}"
#     ds_diff.attrs['epoch2'] = f"{list_year_ranges[epoch2][0]}-{list_year_ranges[epoch2][1]}"
# 
#     # Save the difference dataset and KS statistics to NetCDF files
#     suffix_file = f"_{varname}_{str_event}_{int(qtl_LE*100)}pct_diff{ds_diff.attrs['epoch2']}_{ds_diff.attrs['epoch1']}_CRCM5_{no_membs}membs.nc"
#     diff_file = f"{output_dir}analogues-{var_analogues}_difference{suffix_file}"
#     ks_file = f"{output_dir}analogues-{var_analogues}_KS-statistics{suffix_file}"
#     if os.path.exists(diff_file):
#         print(f"Difference file already exists: {diff_file}")
#     else:
#         ds_diff.chunk({'lat': 100, 'lon': 100}).to_netcdf(f"{output_dir}analogues-{var_analogues}_difference{suffix_file}")
#         print(f"Saved difference for epoch {epoch1} vs {epoch2}")
#     if os.path.exists(ks_file):
#         print(f"KS statistics file already exists: {ks_file}")
#     else:
#         ks_statistics.chunk({'lat': 100, 'lon': 100}).to_netcdf(f"{output_dir}analogues-{var_analogues}_KS-statistics{suffix_file}")
#         print(f"Saved KS statistics for epoch {epoch1} vs {epoch2}")
# 
# 