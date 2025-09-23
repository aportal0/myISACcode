import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from scipy.stats import ks_2samp
import matplotlib.colors as mcolors
import time
from collections import defaultdict
import xesmf as xe
import cftime
from datetime import datetime


## Functions to load data from ERA5 and CERRA

def load_ERA5_data(varname, freq, timestep, lonlat_bounds, l_anom):
    """Loads ERA5 data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Time variables
    years_data = np.unique(timestep.strftime("%Y"))
    if freq == 'daily':
        timestep = (timestep.normalize() + pd.Timedelta(hours=9)).isoformat()
    # Possible varnames: 'mslp', 'z500'
    data_dir = '/media/alice/Crucial X9/portal/data_CNR/ERA5/'+varname+'/'
    if l_anom:
        files = [os.path.join(data_dir, f'ERA5_{varname}_NH_{freq}_{year}_anom.nc') for year in years_data]
    else:
        files = [os.path.join(data_dir, f'ERA5_{varname}_NH_{freq}_{year}.nc') for year in years_data]
    # Load data
    # select variable and timestep
    if varname == 'z500':
        datasets = [xr.open_dataset(file)['z'].sel(time=timestep, plev=50000) / 9.81 for file in files]
        str_lon, str_lat = ('lon', 'lat')
    elif varname == 'mslp':
        datasets = [xr.open_dataset(file)['msl'].sel(time=timestep) * 0.01 for file in files]
        str_lon, str_lat = ('longitude', 'latitude')
    # concatenate datasets along "time"
    data = xr.concat(datasets, dim="time").squeeze()
    # Define and select lon lat masks
    lon_mask, lat_mask = lonlat_mask(data[str_lon].values, data[str_lat].values, lonlat_bounds)
    mask = lat_mask[:, np.newaxis] & lon_mask
    data = data.where(mask, np.nan).dropna(dim=str_lat, how="all").dropna(dim=str_lon, how="all")
    return data


def load_ERA5_clim(varname, doy, lonlat_bounds, l_smoothing):
    """Loads ERA5 climatology 1985-2019 for a given variable and day-of-year (doy) in lonlat_bounds.
    l_smoothing = True for 31-day smoothing time window."""
    # Possible varnames: 'msl'
    # data_dir = '/work_big/users/clima/portal/ERA5/'+varname+'/climatology/'
    data_dir = '/media/alice/Crucial X9/portal/data_CNR/ERA5/'+varname+'/climatology/'
    if l_smoothing:
        file = os.path.join(data_dir, f'ERA5_{varname}_NH_daily_clim_2004-2023_sm31d.nc')
    else:
        file = os.path.join(data_dir, f'ERA5_{varname}_NH_daily_clim_2004-2023.nc')
    # Load data
    # select variable and timestepg
    if varname == 'z500':
        data = xr.open_dataset(file)['z'].sel(dayofyear=doy, plev=50000) / 9.81
        str_lon, str_lat = ('lon', 'lat')
    elif varname == 'mslp':
        data = xr.open_dataset(file)['msl'].sel(dayofyear=doy) * 0.01
        str_lon, str_lat = ('longitude', 'latitude')
    # Define and select lon lat masks
    lon_mask, lat_mask = lonlat_mask(data[str_lon].values, data[str_lat].values, lonlat_bounds)
    mask = lat_mask[:, np.newaxis] & lon_mask
    data = data.where(mask, np.nan).dropna(dim=str_lat, how="all").dropna(dim=str_lon, how="all")
    return data

def load_CERRA_italy_data(varname, timestep, lonlat_bounds):
    """Loads CERRA data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Possible varnames: 'precip'
    data_dir = '/mnt/naszappa/CERRA/daily/nc/regular/'
    years_data = np.unique(timestep.strftime("%Y"))
    files = [os.path.join(data_dir, f'precip_{year}_italy_reg10.nc') for year in years_data]
    # select variable and timestep
    if varname == 'precip':
        datasets = [xr.open_dataset(file)['tp'].sel(time=timestep) for file in files]
    # concatenate datasets along "time"
    data = xr.concat(datasets, dim="time").squeeze()
    # Define and select lon lat masks
    lon_mask, lat_mask = lonlat_mask(data.lon.values, data.lat.values, lonlat_bounds)
    mask = lat_mask[:, np.newaxis] & lon_mask
    data = data.where(mask, np.nan).dropna(dim="lat", how="all").dropna(dim="lon", how="all")
    return data


def load_CERRA_precip(timestep, lonlat_bounds):
    """Loads CERRA data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Possible varnames: 'precip'
    # data_dir = '/mnt/naszappa/CERRA/daily/nc/regular/'
    data_dir = '/media/alice/Crucial X9/portal/data_CNR/CERRA/precip'
    years_data = np.unique(timestep.strftime("%Y"))
    files = [os.path.join(data_dir, f'precip_daily_{year}_remapbil-to-05res.nc') for year in years_data]
    # select variable and timestep
    datasets = [xr.open_dataset(file)['tp'].sel(time=timestep, method="nearest") for file in files]
    # concatenate datasets along "time"
    data = xr.concat(datasets, dim="time").squeeze()
    # Define and select lon lat masks
    lon_mask, lat_mask = lonlat_mask(data.lon.values, data.lat.values, lonlat_bounds)
    mask = lat_mask[:, np.newaxis] & lon_mask
    data = data.where(mask, np.nan).dropna(dim="lat", how="all").dropna(dim="lon", how="all")
    return data


## Functions to open member datasets from CRCM5-LE

def get_anomaly_climatology_paths_CRCM5(CRCM5_dir, varname, list_membs, year_range):
    """
    Return anomaly and climatology file paths for a specific variable and epoch (year_range).
    
    Parameters:
        CRCM5_dir (str): Base directory for the files.
        varname (str): Variable name (e.g., 'tas').
        list_membs (list): List of member names (e.g., ['r1i1p1', 'r2i1p1']).
        year_range (tuple): Year range as a tuple (start_year, end_year).
    
    Returns:
        tuple: (memb_files, memb_files_clim) where each is a dict {member: [file_paths]}.
    """
    paths_files = []
    paths_files_clim = []
    years_sel = list(range(year_range[0], year_range[1] + 1))
    prefix_files = f"{varname}-anom" if varname != "zg" else f"{varname}500-anom"
    suffix_files_clim = f"clim{year_range[0]}-{year_range[1]}_sm31d_05res.nc"

    # Build list of anomaly directories
    dirs_files = [
        os.path.join(CRCM5_dir, varname, memb, str(year), 'res05/')
        for year in years_sel for memb in list_membs
    ]
    # Build list of climatology directories
    dirs_files_clim = [
        os.path.join(CRCM5_dir, varname, memb, 'clim/')
        for memb in list_membs
    ]

    # Collect anomaly files
    for dir in dirs_files:
        if os.path.exists(dir):
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                if os.path.isfile(file_path) and file.startswith(prefix_files):
                    paths_files.append(file_path)

    # Collect climatology files
    for dir in dirs_files_clim:
        if os.path.exists(dir):
            for file in os.listdir(dir):
                file_path = os.path.join(dir, file)
                if os.path.isfile(file_path) and file.endswith(suffix_files_clim):
                    paths_files_clim.append(file_path)

    # Organize by member
    memb_files = create_member_file_dict(paths_files, list_membs)
    memb_files_clim = create_member_file_dict(paths_files_clim, list_membs)

    # Sort the paths
    for memb in memb_files_clim:
        memb_files_clim[memb].sort()
    for memb in memb_files:
        memb_files[memb].sort()

    return memb_files, memb_files_clim


def get_anomaly_climatology_paths_CRCM5_bymonth(CRCM5_dir, varname, list_membs, list_times):
    """
    Return anomaly and climatology file paths for a specific variable and epoch (year_range).
    
    Parameters:
        CRCM5_dir (str): Base directory for the files.
        varname (str): Variable name (e.g., 'tas').
        list_membs (list): List of member names (e.g., ['r1i1p1', 'r2i1p1']).
        list_times (list): List of lists containing times for each member, where each time can be a string or datetime object.

    Returns:
        tuple: (memb_files, memb_files_clim) where each is a dict {member: [file_paths]}.
    """
    # File directory
    dir_files = CRCM5_dir + varname + '/'
    # Initialize a list to hold the file paths
    paths_files = []
    paths_files_clim = []
    prefix_files = f"{varname}-anom" if varname != "zg" else f"{varname}500-anom"
    
    for im, member in enumerate(list_membs):
        
        times = list_times[im]
        
        for time in times:  
            
            # Check if time is a string or datetime object
            if isinstance(time, cftime.DatetimeNoLeap):
                time = datetime(time.year, time.month, time.day, hour=0, minute=0, second=0)
            elif isinstance(time, str):
                time = datetime.strptime(time, "%Y-%m-%d  %H:%M:%S")
            elif isinstance(time, datetime):
                pass
            else:
                raise ValueError("Time must be a string or datetime object.")
            
            # Extract year and month from the time
            year = time.year
            month = time.month
            year_range = get_year_range(year)
            
            # Create labels for the file names
            time_label = f"{year:04d}{month:02d}"
            suffix_file = f"_{time_label}_remapbil-to-05res.nc"
            suffix_files_clim = f"{month:02d}_clim{year_range[0]}-{year_range[1]}_sm31d_05res.nc"
            
            # Create the file name
            file_dir = f"{dir_files}{member}/{year}/res05/"
            if os.path.exists(file_dir):
                for file in os.listdir(file_dir):
                    file_path = os.path.join(file_dir, file)
                    if os.path.isfile(file_path) and file.startswith(prefix_files) and file.endswith(suffix_file):
                        paths_files.append(file_path)
            
            # Create the climatology file name
            file_clim_dir = f"{dir_files}{member}/clim/"
            if os.path.exists(file_clim_dir):
                for file in os.listdir(file_clim_dir):
                    file_path = os.path.join(file_clim_dir, file)
                    if os.path.isfile(file_path) and file.endswith(suffix_files_clim):
                        paths_files_clim.append(file_path)
    
    # Organize by member
    paths_files = list(set(paths_files))
    memb_files = create_member_file_dict(paths_files, list_membs)
    paths_files_clim = list(set(paths_files_clim))
    memb_files_clim = create_member_file_dict(paths_files_clim, list_membs)

    # Sort the paths
    for memb in memb_files_clim:
        memb_files_clim[memb].sort()
    for memb in memb_files:
        memb_files[memb].sort()

    return memb_files, memb_files_clim


def get_precipitation_paths_CRCM5(CRCM5_dir, list_membs, year_range):
    """
    Return anomaly and climatology file paths for a specific variable and epoch (year_range).
    
    Parameters:
        CRCM5_dir (str): Base directory for the files.
        list_membs (list): List of member names (e.g., ['kba', 'kbb']).
        year_range (tuple): Year range as a tuple (start_year, end_year).
    Returns:
        dict: A dictionary where keys are member names and values are lists of file paths.
    """
    varname = 'pr'
    # Define the prefix for files based on year range
    if year_range[1] < 2006:
        prefix_files = [f"{varname}_daysum_historical_"]
    elif year_range[0] >= 2006:
        prefix_files = [f"{varname}_daysum_rcp85_"]
    else:
        prefix_files = [f"{varname}_daysum_historical_", f"{varname}_daysum_rcp85_"]

    # Build file directory
    dir_files = CRCM5_dir + varname + '/'

    # Collect files
    paths_files = []
    if os.path.exists(dir_files):
        for file in os.listdir(dir_files):
            file_path = os.path.join(dir_files, file)
            if (
                os.path.isfile(file_path)
                and file.startswith(tuple(prefix_files))
                # and any(file.startswith(prefix) for prefix in prefix_files)
                and any(member in file for member in list_membs)
            ):
                paths_files.append(file_path)

    # Organize by member
    memb_files = create_member_file_dict(paths_files, list_membs)
    # Sort the paths
    for memb in memb_files:
        memb_files[memb].sort()

    return memb_files


def get_precipitation_paths_CRCM5_bymonth(CRCM5_dir, list_membs, list_times):
    """
    Return anomaly and climatology file paths for a specific variable and epoch (year_range).
    
    Parameters:
        CRCM5_dir (str): Base directory for the files.
        list_membs (list): List of member names (e.g., ['kba', 'kbb']).
        list_times (list): List of lists containing times for each member, where each time can be a string or datetime object.
    Returns:
        dict: A dictionary where keys are member names and values are lists of file paths.
    """
    varname = 'pr'
    # File directory
    dir_files = CRCM5_dir + varname + '/'
    # Initialize a list to hold the file paths
    paths_files = []
    for im, member in enumerate(list_membs):
        times = list_times[im]
        for time in times:  
            # Check if time is a string or datetime object
            if isinstance(time, cftime.DatetimeNoLeap):
                time = datetime(time.year, time.month, time.day, hour=0, minute=0, second=0)
            elif isinstance(time, str):
                time = datetime.strptime(time, "%Y-%m-%d  %H:%M:%S")
            elif isinstance(time, datetime):
                pass
            else:
                raise ValueError("Time must be a string or datetime object.")
            # Extract year and month from the time
            year = time.year
            month = time.month
            # Create labels for the file names
            if year < 2006:
                simulation = 'historical'
            else:
                simulation = 'rcp85'
            time_label = f"{year:04d}{month:02d}"
            # Create the file name
            file = f"{dir_files}{member}/{year}/{varname}_daysum_{simulation}_{member}_remapbil-to-05res_{time_label}.nc"
            # Append the file path to the list
            paths_files.append(file)

    # Organize by member
    memb_files = create_member_file_dict(paths_files, list_membs)
    # Sort the paths
    for memb in memb_files:
        memb_files[memb].sort()

    return memb_files


def create_member_file_dict(paths_files, list_membs):
    """
    Create a dictionary mapping members to their respective file paths.

    Parameters:
    - paths_files: List of file paths.
    - list_membs: List of member names.

    Returns:
    - A defaultdict where keys are member names and values are lists of file paths.
    """
    memb_files = defaultdict(list)
    for path in paths_files:
        for memb in list_membs:
            if memb in path:
                memb_files[memb].append(path)
                break  # Assuming one member per file path
    # return with keys in alphabetical order
    return {m: memb_files[m] for m in sorted(list_membs)}


def open_member_datasets(memb_files_dict, combine, expand_member_dim):
    """
    Open datasets for each member and optionally add 'member' as a dimension.

    Parameters:
    ----------
    memb_files_dict : dict
        Dictionary mapping each member (str) to a list of file paths.
    combine : str
        Combine method for xr.open_mfdataset (default: 'by_coords').
    expand_member_dim : bool
        Whether to add 'member' as a new dimension using expand_dims.
    
    Returns:
    -------
    list of xarray.Dataset
        List of datasets, one for each member with member dimension added.
    """
    list_ds = []
    for memb, files in memb_files_dict.items():
        print(f"Opening files for member: {memb}")
        ds = xr.open_mfdataset(files, combine=combine)
        if expand_member_dim:
            ds = ds.expand_dims(member=[memb])
        list_ds.append(ds)
    return list_ds


def get_year_range(year):
    """
    Get the year range for a given year.
    """
    ranges = [(1955, 1974), (2004, 2023), (2030, 2049), (2080, 2099)]
    year_range = next(([start, end] for start, end in ranges if start <= year <= end), None)
    return year_range


## Function to create lon-lat mask (event-wise)

def box_event_PrMax_alertregions(no_node, no_event):
    """
    Lon-lat box of event selected based on above99 prec over alert regions.
    box_event = [lon_min, lon_max, lat_min, lat_max]
    """
    if no_node == 6 and no_event == 1:
        box_event = [3, 22, 35, 50]
    elif no_node == 1 and no_event == 1:
        box_event = [-5, 20, 31, 50]
    return box_event


def lonlat_mask(lon, lat, lonlat_bounds):
    """Returns the mask for the lonlat_bounds."""
    lon_mask = (lon >= lonlat_bounds[0]) & (lon <= lonlat_bounds[1])
    lat_mask = (lat >= lonlat_bounds[2]) & (lat <= lonlat_bounds[3])
    return lon_mask, lat_mask

## Function to retrieve best model analogue info 
def get_best_model_analogue_info(no_node, no_event, var_analogues):
    """
    Returns the best model analogue info for a given selecion of node and event.
    
    Parameters:
        l_sel (str): Selection type (e.g., 'Italy', 'wide-region', 'alert-regions').
        no_node (int): Node number.
        no_event (int): Event number.
    
    Returns:
        dictionary with "member", "distance" and "date" of the best model analogue.
    """
    if no_node==1 and no_event==1:
        if var_analogues == 'psl':
            dict_best_analogue = {'member': 'kbw', 'distance': 125.24177, 'date': '2019-11-25'}
    
    return dict_best_analogue


## Function to regrid data using xESMF

def regrid_with_xesmf(field_event, box_event, resolution=0.5):
    # Determine input lat/lon names
    lat_name = 'lat' if 'lat' in field_event.dims else 'latitude'
    lon_name = 'lon' if 'lon' in field_event.dims else 'longitude'

    # Define the new grid
    new_lat = np.arange(box_event[2], box_event[3] + resolution, resolution)
    new_lon = np.arange(box_event[0], box_event[1] + resolution, resolution)
    target_grid = xr.Dataset({
        'lat': (['lat'], new_lat),
        'lon': (['lon'], new_lon),
    })

    # Prepare the regridder
    regridder = xe.Regridder(field_event, target_grid, method='bilinear', periodic=False, reuse_weights=False)

    # Regrid and preserve name/attrs
    regridded = regridder(field_event)
    regridded.name = field_event.name  # preserve variable name
    regridded.attrs = field_event.attrs  # optionally keep metadata

    return regridded


## Function for Kolmogorov-Smirnov test

def ks_stat_and_pval(x, y):
    """Perform the Kolmogorov-Smirnov test and return the statistic and p-value."""
    # Remove NaNs from both arrays
    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]
    
    # Edge case: if either array is empty after removing NaNs
    if len(x_clean) == 0 or len(y_clean) == 0:
        return np.array([np.nan, np.nan])
    
    result = ks_2samp(x_clean, y_clean)
    return np.array([result.statistic, result.pvalue])

## Functions to plot data

def plot_geopotential_and_mslp(ax, timestep, lonlat_bounds, z500, msl):
    """Plots the geopotential height and mean sea level pressure data for a given timestep."""
    
    # Create grid 
    lon = z500.lon.values
    lat = z500.lat.values
    # Add cyclic point
    z500, lon1 = add_cyclic_point(z500, coord=lon)
    msl, _ = add_cyclic_point(msl, coord=lon)

    # Plot the data
    ax.coastlines()
    ax.set_extent(lonlat_bounds, crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False)
    # Z500
    z500_levels =  np.arange((z500.min()//100)*100, (z500.max()//100)*100+2*100, 100)
    shade = ax.contourf(lon1, lat, z500, transform=ccrs.PlateCarree(), cmap='viridis', levels=z500_levels)
    cbar = plt.colorbar(shade, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Geopotential height [m]')
    # MSLP
    cont = ax.contour(lon1, lat, msl, transform=ccrs.PlateCarree(), colors='red', linewidth=2, levels=np.arange(900, 1100, 5))
    ax.clabel(cont, inline=True, fontsize=8, fmt="%.0f")
    # Title and labels
    ax.set_title(f"Timestep: {timestep}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')


def plot_precipitation(ax, timestep, lonlat_bounds, precip, precip_levels):
    """Plots the precipitation data for a given timestep."""

    # Create grid 
    lon = precip.lon.values
    lat = precip.lat.values
    # Add cyclic point
    precip, lon1 = add_cyclic_point(precip, coord=lon)

    # Plot the data
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent(lonlat_bounds, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    # Parameters for colormap
    cmap = plt.get_cmap("YlGnBu")
    norm = mcolors.BoundaryNorm(boundaries=precip_levels, ncolors=cmap.N, extend='max')
    # Precipitation
    mesh = ax.pcolormesh(lon1, lat, precip, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, boundaries=precip_levels, ticks=precip_levels, extend='max')
    cbar.set_label('24h precipitation [mm]')
    # Title and labels
    ax.set_title(f"Timestep: {timestep}")


def plot_anom_event(ax, varname, lon, lat, anom_event, clim):
    """Plots the anomaly and DOY climatology for a given event (mslp or z500)."""

    # Set intervals and levels
    if varname == 'z500':
        cbar_int = 50
        levels_clim = np.arange(5000, 6000, 25)
    elif varname == 'mslp':
        cbar_int = 2
        levels_clim = np.arange(950, 1050, 1)
    # Calculate the min and max values around zero for centering
    vmin = np.nanmin(anom_event)
    vmax = np.nanmax(anom_event)
    cbar_center = max(abs(vmin), abs(vmax)) // cbar_int * cbar_int + cbar_int
    cbar_levels = np.arange(-cbar_center, cbar_center+cbar_int, cbar_int)

    # Plot data
    cf = ax.contourf(lon, lat, anom_event, transform=ccrs.PlateCarree(), cmap="RdBu_r", levels= cbar_levels)
    contours = ax.contour(lon, lat, clim, transform=ccrs.PlateCarree(), levels=levels_clim, colors="black", linewidths=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    # Add coastlines
    ax.coastlines(linewidth=1.5)
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    # Add colorbar
    if varname == 'z500':
        cbar_label = "$\\Delta$Z500 (m)"
    elif varname == 'mslp':
        cbar_label = "$\\Delta$mslp (hPa)"
    plt.colorbar(cf, ax=ax, shrink=0.6, label=cbar_label)
    return ax
