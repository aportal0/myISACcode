import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import time


def load_ERA5_data(varname, timestep, lonlat_bounds):
    """Loads ERA5 data for a given variable, timestep and in lonlat_bounds.
    Timestep can be a single datetime object or a list of datetime objects."""
    # Possible varnames: 'msl', 'z500'
    data_dir = '/work_big/users/ghinassi/ERA5/'+varname+'/'
    years_data = np.unique(timestep.strftime("%Y"))
    files = [os.path.join(data_dir, f'ERA5_{varname}_6hr_{year}.nc') for year in years_data]
    # Load data
    # select variable and timestep
    if varname == 'z500':
        datasets = [xr.open_dataset(file)['z'].sel(valid_time=timestep, pressure_level=500) / 9.81 for file in files]
    elif varname == 'msl':
        datasets = [xr.open_dataset(file)['msl'].sel(time=timestep) * 0.01 for file in files]
    # concatenate datasets along "time"
    data = xr.concat(datasets, dim="time").squeeze()
    # Define and select lon lat masks
    lon_mask, lat_mask = lonlat_mask(data.longitude.values, data.latitude.values, lonlat_bounds)
    mask = lat_mask[:, np.newaxis] & lon_mask
    data = data.where(mask, np.nan).dropna(dim="latitude", how="all").dropna(dim="longitude", how="all")
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

def lonlat_mask(lon, lat, lonlat_bounds):
    """Returns the mask for the lonlat_bounds."""
    if lonlat_bounds[0]<0 and lonlat_bounds[1]>0:
        lon_mask = (lon >= lonlat_bounds[0]) | (lon <= lonlat_bounds[1])
    else:
        lon_mask = (lon >= lonlat_bounds[0]) & (lon <= lonlat_bounds[1])
    lat_mask = (lat >= lonlat_bounds[2]) & (lat <= lonlat_bounds[3])
    return lon_mask, lat_mask

def plot_geopotential_and_mslp(ax, timestep, lonlat_bounds, z500, msl):
    """Plots the geopotential height and mean sea level pressure data for a given timestep."""
    
    # Create grid 
    lon = z500.longitude.values
    lat = z500.latitude.values
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