## Script to create daily mean files from 3h resolution files across CRCM5-LE
import numpy as np
import xarray as xr
import subprocess
import multiprocessing
from multiprocessing import Pool
import time
# my function files
import functions_preprocessing as fp

## Parameters
# Variable
var = 'tas'
# Period
year_range = [1955,2099]
# Number of ensemble members
memb_range = [48,50]
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
# Number of processors
nproc = 32

## Computation
# Loop over ensemble members, years, months
# st = time.time()
for memb in range(memb_range[0],memb_range[1]+1):
    # Generate list strings
    files_3h = [
        fp.path_file_CRCM5(var, memb, year, mon, time_res='3h')
        for year in range(year_range[0], year_range[1] + 1)
        for mon in range(mon_range[0], mon_range[1] + 1)
    ]
    files_daily = [
        fp.path_file_CRCM5(var, memb, year, mon, time_res='daily')
        for year in range(year_range[0], year_range[1] + 1)
        for mon in range(mon_range[0], mon_range[1] + 1)
    ]
    nfiles = len(files_3h)
    # Compute and save daily mean (parallelized)
    inputs_funct = zip(files_3h, files_daily, 
                       ['tas']*nfiles, [encoding_netcdf]*nfiles)
    with Pool(nproc) as p:
        p.starmap(fp.save_daily_mean,inputs_funct) ## Pool(12) 175 sec
    
    # Bash remove file command (after checking if daily file is present)
    ifile = 0
    while (ifile < len(files_3h)):
        status = subprocess.call("test -e '{}'".format(files_daily[ifile]), shell=True)
        if status == 0:
            command = f'rm '+files_3h[ifile]
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            fp.check_subprocess(result, memb, year_range[0]+ifile//12, ifile%12+1)
        else:
            print('3h file of member '+memb+', year '+str(year_range[0]+ifile//12)+', month '+str(ifile%12+1)+
                  ' not removed because daily file is missing')
        ifile += 1
# en = time.time()
# print('Time calculation: ',en-st)
