import numpy as np
import xarray as xr
import subprocess
import os
import time


def check_subprocess(result_subprocess, member, year, month):
    """ Function to check result of subprocess """
    # Check if the command was successful
    if result_subprocess.returncode == 0:
        print(f'Successfully removed 3h file of ensemble member {member} in year {int(year)}, month {int(month)}')
    else:
        print(f'Error removing file: {result_subprocess.stderr}')


def save_daily_mean(file_in, file_out, namevar, encoding):
    """ Function takes computes and saves the daily mean values of variable.
    Args:
    file_in (str): input file.
    file_out (str): output file.
    namevar (str): variable on which to compute daily mean.
    encoding (dict): encoding options for saving the dataset.
    """
    # Load the dataset
    ds = xr.open_dataset(file_in)
    # Resample to daily means
    daily_means = ds[namevar].resample(time='1D').mean(dim='time')
    # Adapt dataset
    ds_daily = ds.isel(time=slice(0,len(daily_means['time'].values))).assign_coords(time=daily_means.time)
    ds_daily[namevar] = daily_means
    # Save daily means dataset 
    ds_daily.to_netcdf(file_out, encoding=encoding)
    return


def compute_and_save_daily_mean(file_in, file_out, namevar, encoding):
    """ Function takes computes and saves the daily mean values of variable.
    Args:
    file_in (str): input file.
    file_out (str): output file.
    namevar (str): variable on which to compute daily mean.
    encoding (dict): encoding options for saving the dataset.
    """
    # Load the dataset
    ds = xr.open_dataset(file_in)
    # Resample to daily means
    ds_daily = ds[namevar].resample(time='1D').mean()
    # Save daily means dataset 
    ds_daily.to_netcdf(file_out, encoding=encoding)
    return


def path_file_CRCM5(namevar, memb, year, month, time_res):
    """ Function to generate the path of the CRCM5 files.
    Args:
    namevar (str): variable name.
    memb (int): number of ensemble member (1-50).
    year (int): year.
    month (int): month.
    """
    # Model data directory 
    data_dir = '/work_big/users/portal/CRCM5-LE/'
    # Run type
    if year >= 2006:
        run_type = 'rcp85'
    else:
        run_type = 'historical'
    # Ensemble members
    membs = [
        'kba', 'kbb', 'kbc', 'kbd', 'kbe', 'kbf', 'kbg', 'kbh', 'kbi', 'kbj', 'kbk', 'kbl', 'kbm',
        'kbn', 'kbo', 'kbp', 'kbq', 'kbr', 'kbs', 'kbt', 'kbu', 'kbv', 'kbw', 'kbx', 'kby', 'kbz',
        'kca', 'kcb', 'kcc', 'kcd', 'kce', 'kcf', 'kcg', 'kch', 'kci', 'kcj', 'kck', 'kcl',
        'kcm', 'kcn', 'kco', 'kcp', 'kcq', 'kcr', 'kcs', 'kct', 'kcu', 'kcv', 'kcw', 'kcx'
        ]
    # Model runs
    runs = [
        'r1-r1i1p1', 'r1-r2i1p1', 'r1-r3i1p1', 'r1-r4i1p1', 'r1-r5i1p1', 'r1-r6i1p1', 'r1-r7i1p1', 'r1-r8i1p1',
        'r1-r9i1p1', 'r1-r10i1p1', 'r2-r1i1p1', 'r2-r2i1p1', 'r2-r3i1p1', 'r2-r4i1p1', 'r2-r5i1p1',
        'r2-r6i1p1', 'r2-r7i1p1', 'r2-r8i1p1', 'r2-r9i1p1', 'r2-r10i1p1', 'r3-r1i1p1', 'r3-r2i1p1',
        'r3-r3i1p1', 'r3-r4i1p1', 'r3-r5i1p1', 'r3-r6i1p1', 'r3-r7i1p1', 'r3-r8i1p1', 'r3-r9i1p1',
        'r3-r10i1p1', 'r4-r1i1p1', 'r4-r2i1p1', 'r4-r3i1p1', 'r4-r4i1p1', 'r4-r5i1p1', 'r4-r6i1p1',
        'r4-r7i1p1', 'r4-r8i1p1', 'r4-r9i1p1', 'r4-r10i1p1', 'r5-r1i1p1', 'r5-r2i1p1', 'r5-r3i1p1',
        'r5-r4i1p1', 'r5-r5i1p1', 'r5-r6i1p1', 'r5-r7i1p1', 'r5-r8i1p1', 'r5-r9i1p1', 'r5-r10i1p1'
        ]
    imemb = memb-1
    path_file = data_dir+namevar+'/'+membs[imemb]+'/'+str(year)+'/'+namevar+'_EUR-11_CCCma-CanESM2_'+run_type+'_'+runs[imemb]+'_OURANOS-CRCM5_'+membs[imemb]+'_'+time_res+'_'+str(year)+f"{month:02d}"+'.nc'
    return path_file


def path_folder_CRCM5(namevar, memb, year):
    """ Function to generate the path of the CRCM5 year folders.
    Args:
    namevar (str): variable name.
    memb (int): number of ensemble member (1-50).
    year (int): year.
    month (int): month.
    """
    # Model data directory 
    data_dir = '/work_big/users/portal/CRCM5-LE/'
    # Ensemble members
    membs = [
        'kba', 'kbb', 'kbc', 'kbd', 'kbe', 'kbf', 'kbg', 'kbh', 'kbi', 'kbj', 'kbk', 'kbl', 'kbm',
        'kbn', 'kbo', 'kbp', 'kbq', 'kbr', 'kbs', 'kbt', 'kbu', 'kbv', 'kbw', 'kbx', 'kby', 'kbz',
        'kca', 'kcb', 'kcc', 'kcd', 'kce', 'kcf', 'kcg', 'kch', 'kci', 'kcj', 'kck', 'kcl',
        'kcm', 'kcn', 'kco', 'kcp', 'kcq', 'kcr', 'kcs', 'kct', 'kcu', 'kcv', 'kcw', 'kcx'
        ]
    imemb = memb-1
    path_folder = data_dir+namevar+'/'+membs[imemb]+'/'+str(year)+'/'
    return path_folder


def remove_list2_if_list1_exists(list1, list2):
    # Check if all files in list1 exist and return logical value
    if all(os.path.isfile(file) for file in list1):
        # If all files exist, remove files in list2
        for file in list2:
            if os.path.isfile(file):
                os.remove(file)
        l_rm = True
    else:
        l_rm = False
    return l_rm
