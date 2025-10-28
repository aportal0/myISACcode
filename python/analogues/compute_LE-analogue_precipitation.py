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


# --- Load LE analogue data for all epochs ---
# Anomalies and climatology
# List file paths
list_ds_pr = []  # Initialize an empty list to store data sets for each epoch
analogue_numbers = np.arange(1, no_analogues_LE+1)  # Create an array of analogue numbers
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
        # Select the anomaly and climatology datasets for the current member
        pr_memb = pr_sel[im].sel(time=pr_sel[im].time.isin(times_analogues))
        # Assign analogue numbers to the time coordinate    
        no_analogues = len(pr_memb.time)  # Number of analogues for the current member
        analogue_numbers = np.arange(1, no_analogues+1)  # Create an array of analogue numbers for the current member
        pr_memb = pr_memb.assign_coords(time=analogue_numbers)
        pr_analogues.append(pr_memb)
    
    # Concatenate the datasets for the current epoch
    ds_pr_analogues = xr.concat(pr_analogues, dim='member')[varname]
    ds_pr_analogues = ds_pr_analogues.rename({"time": "analogue"})

    # Select event box
    if i == 0:  # For the first epoch, select the box from the event
        lon_mask, lat_mask = fanPM.lonlat_mask(ds_pr_analogues.lon.values, ds_pr_analogues.lat.values, box_event)
        mask = lat_mask[:, np.newaxis] & lon_mask
        mask_xr = xr.DataArray(
            mask,
            dims=["lat", "lon"],
            coords={"lat": ds_pr_analogues.lat.values, "lon": ds_pr_analogues.lon.values},
        )
    ds_pr_analogues = ds_pr_analogues.where(mask_xr, drop=True)

    # Save in list by epoch
    list_ds_pr.append(ds_pr_analogues)
        
# # Print dataset for the first epoch
# print(f"Anomaly dataset for epoch {list_year_ranges[0]}: {list_ds_anom[0]}")
# print(f"Climatology dataset for epoch {list_year_ranges[0]}: {list_ds_clim[0]}")


# --- Kolmogorov-Smirnov test for significance ---

# Perform the Kolmogorov-Smirnov test for each pair of epochs
list_ks_stats = []  # Initialize an empty list to store the KS statistics
for i, (epoch1, epoch2) in enumerate(diff_indices):
    print(f"Computing KS test for epoch {epoch1} vs epoch {epoch2}")
    members = np.arange(0, no_membs)  # Select all members for the KS test
    # Get the datasets for the two epochs
    ds_epoch1 = list_ds_pr[epoch1].isel(member=members)
    ds_epoch2 = list_ds_pr[epoch2].isel(member=members)
    
    ds1_flat = ds_epoch1.stack(analogue_all=('member', 'analogue')).chunk({'analogue_all': -1})
    ds2_flat = ds_epoch2.stack(analogue_all=('member', 'analogue')).chunk({'analogue_all': -1})

    # Perform the Kolmogorov-Smirnov test for each grid point
    ks_statistics = xr.apply_ufunc(
        fanPM.ks_stat_and_pval,
        ds1_flat,
        ds2_flat,
        input_core_dims=[['analogue_all'], ['analogue_all']],
        output_core_dims=[['output']],
        output_sizes={"output": 2},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
    )
    ks_statistics = ks_statistics.assign_coords(output=["diff_statistic", "pvalue"])
    # Add the datasets to the lists
    list_ks_stats.append(ks_statistics)


# --- Compute LE analogue differences ---
# Compute differences between LE analogues from different epochs
list_ds_diff = []  # Initialize an empty list to store the differences
for i, (epoch1, epoch2) in enumerate(diff_indices):
    # Compute the difference between the two epochs
    ds_epoch1_mean = (list_ds_pr[epoch1]).mean(dim=('member','analogue'))
    ds_epoch2_mean = (list_ds_pr[epoch2]).mean(dim=('member','analogue'))
    ds_diff = ds_epoch2_mean - ds_epoch1_mean
    # Add epoch information to the dataset
    ds_diff.attrs['epoch1'] = f"{list_year_ranges[epoch1][0]}-{list_year_ranges[epoch1][1]}"
    ds_diff.attrs['epoch2'] = f"{list_year_ranges[epoch2][0]}-{list_year_ranges[epoch2][1]}"
    # Add the dataset to the lists
    list_ds_diff.append(ds_diff)


# --- Save to NetCDF files ---
if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

# Save each difference and KS-stat dataset to a NetCDF file
for i in range(len(list_ds_diff)):
    ds_diff = list_ds_diff[i].chunk({'lon': -1, 'lat': -1})
    ks_stats = list_ks_stats[i].chunk({'lon': -1, 'lat': -1})
    suffix_file = f"_{varname}_{str_event}_{int(qtl_LE*100)}pct_diff{ds_diff.attrs['epoch2']}_{ds_diff.attrs['epoch1']}_CRCM5_{no_membs}membs.nc"

    # Save the difference dataset
    diff_file = f"{output_dir}analogues-{var_analogues}_difference{suffix_file}"
    if not os.path.exists(diff_file):
        ds_diff.to_netcdf(diff_file)
        print(f"Saved difference dataset to {diff_file}")
    # Save the KS statistics dataset
    stat_file = f"{output_dir}analogues-{var_analogues}_KS-statistics{suffix_file}"
    if not os.path.exists(stat_file):
        ks_stats.to_netcdf(stat_file)
        print(f"Saved KS statistics dataset to {stat_file}")
del list_ds_diff
del list_ks_stats

# Save absolute value by epoch to NetCDF files
for i, year_range in enumerate(list_year_ranges):
    pr_epoch = list_ds_pr[i].chunk({'lon': -1, 'lat': -1})
    pr_map = pr_epoch.mean(dim=('member','analogue'))
    suffix_file = f"{varname}_{str_event}_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5_{no_membs}membs.nc"

    # Compute mean precipitation in mask
    if event_origin == 'ERA5':
        mask = xr.open_dataset(f'./analogue_data/event_data/{varname}-mask_{str_event}_CERRA.nc')
    elif event_origin == 'CRCM5-LE':
        mask = xr.open_dataset(f'./analogue_data/BAM_data/{varname}-mask_BAM-{var_analogues}_{str_event}_OND_2004-2023_CRCM5-LE_49membs.nc')
    pr_masked = pr_epoch.where(mask['pr_mask']==1)  # Apply mask to the precipitation data
    # weights = cos(lat)
    weights = np.cos(np.deg2rad(pr_epoch['lat']))
    weights = weights.broadcast_like(pr_epoch)   # expand to lat/lon grid
    # now do a weighted mean over the region
    pr_regional_mean = pr_masked.weighted(weights).mean(dim=("lat","lon")).chunk({'analogue': -1, 'member': -1}).squeeze()

    # Save the absolute value dataset
    pr_file = f"{output_dir}analogues-{var_analogues}_{suffix_file}"
    if not os.path.exists(pr_file):
        pr_map.to_netcdf(pr_file)
        print(f"Saved anomaly dataset to {pr_file}")
    # Save the absolute value dataset with regional mean
    pr_regional_file = f"{output_dir}analogues-{var_analogues}_mask-mean-{suffix_file}"
    if not os.path.exists(pr_regional_file):
        pr_regional_mean.to_netcdf(pr_regional_file)
        print(f"Saved regional mean dataset to {pr_regional_file}")