#! 

data_dir='/work_big/users/ghinassi/ERA5/msl/'
work_dir='/work_big/users/portal/ERA5/msl/climatology/'
ls ${data_dir}ERA5_msl_6hr_{1985..2019}.nc > ${work_dir}filelist.txt

# Compute daily climatology
cdo mergetime -f nc ${work_dir}filelist.txt ${work_dir}merged.nc
cdo ydaymean ${work_dir}merged.nc ${work_dir}climatology_1985-2019.nc

# Extract the first and last 15 days of the year and extend the climatology
cdo selday,-15/-1 ${work_dir}climatology_1985-2019.nc ${work_dir}end_days.nc
cdo selday,1/15 ${work_dir}climatology.nc ${work_dir}start_days.nc
cdo cat ${work_dir}end_days.nc ${work_dir}climatology_1985-2019.nc ${work_dir}start_days.nc ${work_dir}extended_climatology.nc

# Compute the running mean
cdo runmean,31 ${work_dir}extended_climatology.nc ${work_dir}running_mean.nc

# Trim to original year length
cdo selday,1/365 ${work_dir}running_mean.nc ${work_dir}running_mean_climatology_1985-2019.nc

# Clean up
rm ${work_dir}merged.nc ${work_dir}end_days.nc ${work_dir}start_days.nc ${work_dir}extended_climatology.nc ${work_dir}running_mean.nc
