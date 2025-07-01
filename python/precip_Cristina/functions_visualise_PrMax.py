import numpy as np
import xarray as xr
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import time
from collections import defaultdict


## Functions to load data from ERA5 and CERRA

def load_ERA5_data(varname, freq, timestep, lonlat_bounds, l_anom):
    """Loads ERA5 data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Time variables
    years_data = np.unique(timestep.strftime("%Y"))
    if freq == 'daily':
        timestep = (timestep.normalize() + pd.Timedelta(hours=9)).isoformat()
    # Possible varnames: 'mslp', 'z500'
    data_dir = '/work_big/users/clima/portal/ERA5/'+varname+'/'
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
    data_dir = '/work_big/users/clima/portal/ERA5/'+varname+'/climatology/'
    if l_smoothing:
        file = os.path.join(data_dir, f'ERA5_{varname}_NH_daily_clim_2004-2023_sm31d.nc')
    else:
        file = os.path.join(data_dir, f'ERA5_{varname}_NH_daily_clim_2004-2023.nc')
    # Load data
    # select variable and timestep
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

def load_CERRA_data(varname, timestep, lonlat_bounds):
    """Loads CERRA data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Possible varnames: 'precip'
    data_dir = '/work_big/users/zappa/CERRA-LAND/daily/nc/regular/'
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


## Function to open member datasets from CRCM5-LE

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
    return memb_files

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


def plot_precipitation(ax, timestep, lonlat_bounds, precip):
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
    levels = np.arange(0, 210, 20)
    cmap = plt.get_cmap("YlGnBu")
    norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, extend='max')
    # Precipitation
    mesh = ax.pcolormesh(lon1, lat, precip, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)
    cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.05, boundaries=levels, ticks=levels, extend='max')
    cbar.set_label('24h precipitation [mm]')
    # Title and labels
    ax.set_title(f"Timestep: {timestep}")


def plot_anom_event(varname, lon, lat, anom_event, clim):
    """Plots the anomaly and DOY climatology for a given event (mslp or z500)."""

    # Create a plot with Cartopy
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

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
        cbar_label = "$\Delta$Z500 (m)"
    elif varname == 'mslp':
        cbar_label = "$\Delta$mslp (hPa)"
    cbar = fig.colorbar(cf, ax=ax, shrink=0.6, label=cbar_label)
    return fig, ax