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
analogue_dir = './analogue_data/analogue_times_distances/'
output_dir = './analogue_data/analogue_differences/'  # Directory to save the output files
output_mask_dir = './analogue_data/pr_in_mask/'  # Directory to save the regional mean files

if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

if not os.path.exists(output_mask_dir):
    os.makedirs(output_mask_dir)

# --- Event and LE analogue definition ---
# Event
lselect = 'alertregions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 3
no_event = 8

event_origin = 'CRCM5-LE'  # 'ERA5' or 'CRCM5-LE'
if event_origin == 'ERA5':
    str_event = f'node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'
elif event_origin == 'CRCM5-LE':
    str_event = f'BAM-node{no_node}-extreme{no_event}-{lselect}'  # 'Italy' or 'wide-region' or 'alert-regions'

# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)

# Variable
varname = 'pr' # Variable to compute the difference between analogues, e.g. 'zg' for geopotential height
var_analogues = 'psl-zg500-std'  # Variable used to find the analogues, e.g. 'psl' for sea level pressure
var_BAM = 'psl'

# Quantile and analogue spacing
qtl_search = 0.99

# Number of analogues per member (out of 18 ~ 99th pct) and corresponding qtl
no_analogues_LE = 18
qtl_LE = 0.99
if qtl_LE*100 % 1 == 0:
    qtl_LE_str = f"{int(qtl_LE*100)}pct"
else:
    qtl_LE_str = f"{qtl_LE*100:.1f}pct"

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
        file_path = f'{analogue_dir}times_distances_analogues-{var_analogues}_{str_event}_{int(qtl_search*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5-LE_memb-{memb}.npz'
        # Load the data from the npz file
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        times = data['times'][:no_analogues_LE]
        distances = data['distances'][:no_analogues_LE]
        epoch_data[memb] = {'times': times, 'distances': distances}
    ensemble_data.append(epoch_data)


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

        # Compute and save regional mean precipitation in BAM mask
        if mask_pr is None:
            if event_origin == 'ERA5':
                mask_pr = xr.open_dataset(f'./analogue_data/event_data/{varname}-mask_{str_event}_CERRA.nc')
            elif event_origin == 'CRCM5-LE':
                pattern = f'./analogue_data/BAM_data/{varname}-mask_BAM-{var_BAM}_{str_event}_*_2004-2023_CRCM5-LE_49membs.nc'
                mask_files = glob.glob(pattern)
                mask_pr = xr.open_dataset(mask_files[0])  # Assuming there's only one matching file
                town_names, _, town_masks, town_masks_extended = fanPM.create_town_mask(pr_memb.lon.values, pr_memb.lat.values, no_node, no_event)
            # weights = cos(lat)
            weights = np.cos(np.deg2rad(mask_pr['lat']))
            weights = weights.broadcast_like(mask_pr)   # expand to lat/lon grid
            # land mask
            land_mask = fanPM.create_land_mask(mask_pr.lon.values, mask_pr.lat.values)
            land_masks = []
            for itown in range(len(town_names)):
                land_mask_i = (land_mask + town_masks[itown] > 0)
                land_masks.append(land_mask_i)

        # Regional precip
        pr_masked = pr_memb.where(mask_pr['pr_mask']==1)  # Apply mask to the precipitation data
        pr_memb_regional_mean = pr_masked.weighted(weights).mean(dim=("lat","lon")).squeeze() # weighted mean over the region
        # Save regional mean to NetCDF file
        suffix_file = f"{varname}_{str_event}_{qtl_LE_str}_{year_range[0]}-{year_range[1]}_CRCM5_{memb}.nc"
        pr_regional_file = f"{output_mask_dir}analogues-{var_analogues}_mask-mean-{suffix_file}"
        if os.path.exists(pr_regional_file):
            print(f"Regional mean of epoch {i} already exists: {pr_regional_file}")
        else:
            pr_memb_regional_mean[varname].to_netcdf(pr_regional_file)
            print(f"Saved regional mean of epoch {i}, member {memb}")

        # Town precip
        for itown in range(len(town_names)):
            str_town = town_names[itown]
            # Compute spatial averages
            pr_memb_town = pr_memb['pr'].where(town_masks[itown] == 1).dropna(dim="lon", how="all").dropna(dim="lat", how="all").squeeze()  # weighted mean over the town grid-point  # Apply town mask
            pr_town_extended = pr_memb['pr'].where(town_masks_extended[itown] == 1)  # Apply expanded town mask
            pr_town_extended_land = pr_memb['pr'].where((town_masks_extended[itown] == 1) & (land_masks[itown] == 1))  # Apply land mask within the expanded town mask
            pr_memb_town_extended = pr_town_extended.weighted(weights).mean(dim=("lat", "lon")).squeeze()  # weighted mean over the town-centered box
            pr_memb_town_extended_land = pr_town_extended_land.weighted(weights).mean(dim=("lat", "lon")).squeeze()  # weighted mean over the town-centered box and land mask
            # Build dataset
            ds_town = pr_memb_town.to_dataset(name="pr_town")
            ds_town["pr_town_7x7"] = pr_memb_town.copy(data=pr_memb_town_extended)
            ds_town["pr_town_7x7_land"] = pr_memb_town.copy(data=pr_memb_town_extended_land)
            # Save town mean to NetCDF file
            suffix_file = f"{varname}_{str_event}_{qtl_LE_str}_{year_range[0]}-{year_range[1]}_CRCM5_{memb}.nc"
            pr_town_file = f"{output_mask_dir}towns/analogues-{var_analogues}_{str_town}-{suffix_file}"
            if os.path.exists(pr_town_file):
                print(f"{str_town} precip of epoch {i} already exists: {pr_town_file}")
            else:
                ds_town.to_netcdf(pr_town_file)
                print(f"Saved {str_town} precip of epoch {i}")

# BAM Vaia - Pordenone - 45.9613677,12.6466817 lat lon (impact in BAM)
# BAM Vaia - Trieste - 45.6523727,13.7423767 lat lon (impact in analogues)
# BAM ER flood 1996 - Rimini - 44.0535, 12.5334 lat lon (impact in BAM)
# BAM ER flood 1996 - Bologna - 44.4992288,11.2491136 lat lon (impact in analogues)
# BAM Liguria flood 2016 - Imperia - 43.8141665,7.6823612 lat lon (impact in BAM)
# BAM Liguria flood 2016 - Genova - 44.4470705,8.8082602 lat lon (impact in analogues)
# BAM ER flood 28-10-2024 - Pesaro - 43.8998436,12.8738523 lat lon (impact in BAM)
# BAM ER flood 28-10-2024 - Napoli - 40.8540339,14.1640303 lat lon (impact in analogues)
# BAM ER flood 02-05-2023 - Cesena - 44.1492948,12.1798915 lat lon (impact in BAM/analogues)
# BAM ER flood 02-05-2023 - Reggio Emilia - 44.7016831,10.5969829 lat lon (impact in BAM/analogues)

