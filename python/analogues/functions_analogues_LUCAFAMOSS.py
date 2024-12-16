from scipy.spatial.distance import euclidean
import numpy as np
import xarray as xr

# ---------------------------------- #

def function_distance(X, Y):
    
    """
    INPUT:
    - X[lon,lat]       -> Target pattern
    - Y[time,lon,lat]  -> Dataset in which look for the target pattern
    
    OUTPUT:
    - dist[time]       -> Euclidean distance between the target pattern X and Y
    """
    
    x = X.values.flatten()
    y = Y.values.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    
    dist = np.asarray([euclidean(y[time],x) for time in range(y.shape[0])])    
    
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
