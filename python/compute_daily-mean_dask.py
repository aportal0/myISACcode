## Script to create daily mean files from 3h resolution files across CRCM5-LE
import numpy as np
import xarray as xr
import time
import dask
from dask.distributed import LocalCluster, Client, progress
import sys
# my function files
import functions_preprocessing as fp

## Read prompt arguments
var = sys.argv[1]
memb = int(sys.argv[2])
year = int(sys.argv[3])
n_cpu = int(sys.argv[4])

print('Variable:',var)
print('Member:',memb)
print('Year:',year)
print('Number of CPUs:',n_cpu)

## Additional parameters
# Months
mon_range = [1,12]
# Encoding (for compressing netcdf output)
encoding_netcdf = {
    var: {
        'zlib': 'True',  # Use gzip compression
        'complevel': 5,  # Compression level (1-9)
        'dtype': 'float32'      # Optionally change data type to reduce size
    }
}

## Main code
if __name__ == "__main__":
    cluster = LocalCluster(n_workers=n_cpu)
    client = Client(cluster)
    print(client)
    
    # Generate list strings
    files_3h = [
        fp.path_file_CRCM5(var, memb, year, mon, time_res='3h')
        for mon in range(mon_range[0], mon_range[1] + 1)
    ]
    files_daily = [
        fp.path_file_CRCM5(var, memb, year, mon, time_res='daily')
        for mon in range(mon_range[0], mon_range[1] + 1)
    ]
    nfiles = len(files_3h)
    
    # Use Dask to parallelize the loading, processing, and saving of files
    st = time.time()
    dd = [dask.delayed(fp.compute_and_save_daily_mean)(file_in, file_out, var, encoding_netcdf) 
          for file_in, file_out in zip(files_3h, files_daily)]
    dp = dask.persist(*dd)
    dask.compute(*dd)
    print('Time: ',time.time()-st) ## TIME
    client.shutdown()
    cluster.close()
   
    # Romove 3h files
    if fp.remove_list2_if_list1_exists(files_daily, files_3h):
        print(f'3h files memb {memb} year {year} removed successfully')
    else:
        print(f'3h files memb {memb} year {year} not removed because daily files are missing')
    
