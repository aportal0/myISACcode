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
# fig_dir = '/home/portal/figures/case-studies_byNode/'

# alice
CERRA_dir = '/media/alice/Extreme SSD1/folders/data_CNR/CERRA-Land/'
ERA5_dir = '/media/alice/Extreme SSD1/folders/data_CNR/ERA5/'
CRCM5_dir = '/media/alice/Extreme SSD1/folders/data_CNR/CRCM5-LE/'
fig_dir = './figures/'


# --- Event Definition ---

# Event selection
lselect = 'alert-regions'  # 'Italy' or 'wide-region' or 'alert-regions'
no_node = 1
no_event = 1

# Upload the data
if lselect == 'wide-region':
    df_events = pd.read_excel(CERRA_dir+'events_on_wide_regions.xlsx', sheet_name=no_node-1)
elif lselect == 'alert-regions':
    df_events = pd.read_excel(CERRA_dir+'events_cum_on_above99_alert_regions.xlsx', sheet_name=no_node-1)
elif lselect == 'Italy':
    df_events = pd.read_excel(CERRA_dir+'events_on_Italy.xlsx', sheet_name=no_node-1)
time_event = df_events['Time'].iloc[no_event-1] + pd.Timedelta('12h')
doy_event =  time_event.timetuple().tm_yday

# Define lon-lat box of event
box_event = fanPM.box_event_PrMax_alertregions(no_node,no_event)


# --- Parameters LE analogue search ---

# Variable
varname = 'psl'

# Quantile and analogue spacing
qtl_LE = 0.99
analogue_spacing_memb = 7 # days

# Number of ensemble members
no_membs = 2 

# Time
year_range = [1955, 1974] # past [1955-1974], present [2004-2023], near-future [2030-2049], far future [2080-2099]
years_sel = np.arange(year_range[0], year_range[1]+1)

# Months for analogue selection
month_event = time_event.month
months_sel = [month_event-1, month_event, month_event+1]
month_names = [calendar.month_abbr[month] for month in months_sel]
str_months = ''.join([name[0] for name in month_names])


# --- Load event data ---

# From ERA5
dmslp_event = fanPM.load_ERA5_data('mslp', 'daily', time_event, box_event, l_anom=True)
mslp_clim = fanPM.load_ERA5_clim('mslp', doy_event, box_event, l_smoothing=True)

# Regrid the data to the desired resolution
dmslp_event_regridded = fanPM.regrid_with_xesmf(dmslp_event, box_event, resolution=0.5)
mslp_clim_regridded = fanPM.regrid_with_xesmf(mslp_clim, box_event, resolution=0.5)


# --- File paths LE ---

# List of members
list_membs = [name for name in os.listdir(CRCM5_dir + varname) if os.path.isdir(os.path.join(CRCM5_dir + varname, name))]
list_membs = sorted(list_membs)[:no_membs]  # Select the first 'no_membs' members

# Anomalies
# List file paths
dirs_files = [CRCM5_dir + varname + '/' + membs + '/'+ str(year) + '/res05/' for year in years_sel for membs in list_membs]
prefix_files = varname + '-anom'
# Loop through each item in the main folder
paths_files = []
for dir in dirs_files:
    # Loop through files in the subdirectory
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)  # Full file path
        if os.path.isfile(file_path):  # Check if it is a file
            paths_files.append(file_path) if file.startswith(prefix_files) else None
# Create a dictionary: member -> list of files
memb_files = fanPM.create_member_file_dict(paths_files, list_membs)

# Climatology
# List file paths
dirs_files_clim = [CRCM5_dir + varname + '/' + membs + '/clim/' for membs in list_membs]
suffix_files_clim = 'clim'+str(year_range[0])+'-'+str(year_range[1])+'_sm31d_05res.nc'
# Loop through each item in the main folder
paths_files_clim = []
for dir in dirs_files_clim:
    # Loop through files in the subdirectory
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)  # Full file path
        if os.path.isfile(file_path):  # Check if it is a file
            paths_files_clim.append(file_path) if file.endswith(suffix_files_clim) else None
# Create a dictionary: member -> list of files
memb_files_clim = fanPM.create_member_file_dict(paths_files_clim, list_membs)
# Sort paths for each member (by filename or full path)
for memb in memb_files_clim:
    memb_files_clim[memb].sort()


# --- Load LE data ---

# Make list of datasets and add 'member' coordinate
list_ds = fanPM.open_member_datasets(memb_files, combine='by_coords', expand_member_dim=True)
list_ds_clim = fanPM.open_member_datasets(memb_files_clim, combine='by_coords', expand_member_dim=True)
# Concatenate and scale
dmslp_tmp_LE = xr.concat(list_ds, dim='member')[varname] * 0.01
mslp_tmp_clim_LE = xr.concat(list_ds_clim, dim='member')[varname] * 0.01


# --- Prepare data for analogue search ---

# Select time range
dmslp_tmp_LE = dmslp_tmp_LE.sel(time=dmslp_tmp_LE.time.dt.month.isin(months_sel))
dmslp_tmp_LE = dmslp_tmp_LE.sel(time=dmslp_tmp_LE.time.dt.year.isin(years_sel))
mslp_tmp_clim_LE = mslp_tmp_clim_LE.sel(time=mslp_tmp_clim_LE.time.dt.month.isin(months_sel))

# Select event box
lon_mask_LE, lat_mask_LE = fanPM.lonlat_mask(dmslp_tmp_LE.lon.values, dmslp_tmp_LE.lat.values, box_event)
mask_LE = lat_mask_LE[:, np.newaxis] & lon_mask_LE
mask_xr_LE = xr.DataArray(
    mask_LE,
    dims=["lat", "lon"],
    coords={"lat": dmslp_tmp_LE.lat.values, "lon": dmslp_tmp_LE.lon.values},
)
dmslp_sel_LE = dmslp_tmp_LE.where(mask_xr_LE, drop=True)
mslp_sel_clim_LE = mslp_tmp_clim_LE.where(mask_xr_LE, drop=True)

# Fix number of analogues per member based on the quantile
n_analogues_LE = int(np.round(len(dmslp_sel_LE.time) * (1-qtl_LE)))


# --- Compute Euclidean distance ---

# Compute distance to the event
dist_LE = []  # Initialize an empty list to store distances for each member
for memb in list_membs:
    # Compute euclidean distance from the event to the selected mslp data for each member
    dist_memb = fan.function_distance(dmslp_event_regridded, dmslp_sel_LE.sel(member=memb), nan_version=True)
    dist_LE.append(dist_memb)


# --- Search for analogues ---

# Initialize lists to store analogue data
indices_filtered_analogues_LE = []  # Initialize an empty list to store filtered indices for each member
times_filtered_analogues_LE = []  # Initialize an empty list to store filtered analogue times for each member
dist_filtered_analogues_LE = []  # Initialize an empty list to store filtered distances for each member
# Time values
all_times_LE = dmslp_sel_LE.time.values

# Find analogue indices for each member
for ii, memb in enumerate(list_membs):
    print('Processing member:', memb)

    # First search of n_analogues
    factor_0sel = 2
    qtl_0sel = 1 - ((1 - qtl_LE) * factor_0sel)  # First selection of the quantile, for extracting n_analogues
    l_0sel = True  # Flag for selection of analogues
    
    while l_0sel:
        # Compute log-transformed distance
        logdist_memb = np.log(1 / dist_LE[ii])
        # Threshold at given quantile
        thresh_0sel = np.percentile(logdist_memb, qtl_0sel * 100, axis=0)
        mask_analogues_memb = (logdist_memb >= thresh_0sel)
        indices_analogues_memb = np.where(mask_analogues_memb)[0]  # indices of all valid analogue times
    
        # Filter analogues based on the mask and logdist, ensuring they are spaced correctly (analogue_spacing days apart)
        indices_filtered_analogues_memb = fan.timefilter_analogues(indices_analogues_memb, logdist_memb, all_times_LE, analogue_spacing_memb)

        if len(indices_filtered_analogues_memb) >= n_analogues_LE:
            indices_filtered_analogues_memb = indices_filtered_analogues_memb[:n_analogues_LE]
            l_0sel = False
            print("Selection completed using pool data from quantile", qtl_0sel)
        else:
            factor_0sel +=1
            qtl_0sel = 1 - ((1 - qtl_LE) * factor_0sel)
            
    # Save data in lists
    indices_filtered_analogues_LE.append(indices_filtered_analogues_memb)
    times_filtered_analogues_LE.append(all_times_LE[indices_filtered_analogues_memb])
    dist_filtered_analogues_LE.append(dist_LE[ii][indices_filtered_analogues_memb])

# --- Save analogue data ---

# Save the indices of the filtered analogues to a npz file
for im, memb in enumerate(list_membs):
    # Create the file path for each member
    npz_file_path = f'./analogue_data/times_distances_analogues-{varname}_node{no_node}-extreme{no_event}-alertregions_{int(qtl_LE*100)}pct_{year_range[0]}-{year_range[1]}_CRCM5-LE_memb-{memb}.npz'
    np.savez(npz_file_path, 
             times=times_filtered_analogues_LE[im], 
             distances=dist_filtered_analogues_LE[im])