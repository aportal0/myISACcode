from scipy.spatial.distance import euclidean
import numpy as np
import xarray as xr

# ---------------------------------- #

def function_distance(X, Y, nan_version):
    
    """
    INPUT:
    - X[lon,lat]       -> Target pattern
    - Y[time,lon,lat]  -> Dataset in which look for the target pattern
    
    OUTPUT:
    - dist[time]       -> Euclidean distance between the target pattern X and Y
    """
    
    x = X.values.flatten()
    y = Y.values.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    
    if nan_version:
        # If nan_version is True, use the function that handles NaNs
        dist = np.asarray([nan_euclidean(y[time], x) for time in range(y.shape[0])])
    else:
        # Otherwise, use the standard Euclidean distance
        dist = np.asarray([euclidean(y[time], x) for time in range(y.shape[0])])
    
    return dist

# ---------------------------------- #

def function_occurrence(dist,start_date,end_date,quantile):
    
    """
    INPUT:
    - dist[time,members]
    - start_date            -> Start date of reference period
    - end_date              -> End date of reference period
    - quantile              -> The quantile to look for the analogues. 
                               If quantile = 0.98, the script highlights the best 2% of analogues.
                               
    OUTPUT:
    - occurrence[time]      -> Array with 1 in the dates when the analogue occurs
    """

    logdist = np.log(1/dist)
    thresh = np.percentile(logdist.sel(time=slice(start_date,end_date)), quantile*100, axis = 0)
    
    occurrence = xr.DataArray(data=np.ones((dist.shape)),dims=dist.dims,coords=dist.coords)  
    occurrence = occurrence.where(logdist >= thresh)
   
    return occurrence

# ---------------------------------- #

def nan_euclidean(u, v):
    # Ensure u and v are numpy arrays to handle NaNs properly
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Identify NaN indices
    nan_mask = np.isnan(u) | np.isnan(v)
    
    # Replace NaNs with a specific value (e.g., 0 or large number) if desired
    u[nan_mask] = 0
    v[nan_mask] = 0
    
    # Calculate Euclidean distance ignoring NaNs
    return np.sqrt(np.sum((u - v) ** 2))

# ---------------------------------- #

def timefilter_analogues(analogue_indices, logdist, time_sel, time_spacing):
    """
    Filters analogue dates based on a precomputed logdist,
    ensuring analogues are at least ±7 days apart from each other.

    Parameters:
        analogue_indices (np.ndarray): Indices for valid analogues (refers to time_sel) (n_analogues)
        logdist (np.ndarray): Log-transformed distance array (n_times)
        time_sel (xr.DataArray): Range of times in field selection (n_times)
        time_spacing (int): Minimum spacing in days between selected analogues

    Returns:
        selected_indices (np.ndarray): Indices of selected analogue times in time_sel
    """

    # Select best analogues with ≥7-day spacing among themselves
    sorted_indices = analogue_indices[np.argsort(-logdist[analogue_indices])]  # sort by descending logdist (best analogue first)

    selected_indices = []
    selected_times = [] # store already accepted analogue times (for spacing check)

    for idx in sorted_indices:
        this_time = time_sel[idx]
        if all(abs((this_time - t).days) >= time_spacing for t in selected_times):
            selected_indices.append(idx)
            selected_times.append(this_time)
    
    return np.array(selected_indices,dtype=int)