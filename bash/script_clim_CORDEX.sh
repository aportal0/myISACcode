#!/bin/bash

varname=$1
gcm_name=$2
regm_name=$3
memb_name=$4
year_start=$5
year_end=$6

#SBATCH -J remap_CORDEX
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${gcm_name}_${regm_name}_${memb_name}.out
#SBATCH --error=${gcm_name}_${regm_name}_${memb_name}.err
#SBATCH --partition=batch

# Parameters
scriptdir="/home/portal/work/myISACcode/bash/"
window_rmean=31

#-----------------

# Compute climatology and anomalies over range years

# Define file paths
#wdir_hist="/home/portal/work_big/CORDEX/historical/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
#files_hist="${wdir_hist}${varname}_EUR-11_${gcm_name}_historical_${memb_name}_*${regm_name}_v1_day_*_remapbil-to-05res.nc"
wdir="/home/portal/work_big/CORDEX/*/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
files="${wdir_rcp85}${varname}_EUR-11_${gcm_name}_*_${memb_name}_*${regm_name}_v1_day_*_remapbil-to-05res.nc"

# Define tmp and clim file names
wdir_clim="/home/portal/work_big/CORDEX/clim/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
mkdir -p $wdir_clim
file_clim="${wdir_clim}${varname}_EUR-11_${gcm_name}_${memb_name}_*${regm_name}_v1_day_clim${year_start}-${year_end}_remapbil-to-05res.nc"
file_clim_sm="${wdir_clim}${varname}_EUR-11_${gcm_name}_${memb_name}_*${regm_name}_v1_day_clim${year_start}-${year_end}_sm${window_rmean}d_remapbil-to-05res.nc"
file_tmp="${wdir_clim}tmp_clim${year_start}-${year_end}.nc"


# Compute clim and smooth with running mean
cdo -O mergetime -selyear,$year_start/$year_end $files $file_tmp

cdo -O ydaymean $file_tmp $file_clim

cdo cat -selmon,12 "$file_clim" "$file_clim" -selmon,1 "$file_clim" "${file_clim}_ext"

cdo runmean,${window_rmean} "${file_clim}_ext" "${file_clim}_ext_sm"
cdo seltimestep,17/380 "${file_clim}_ext_sm" $file_clim_sm


# Compute anomaly
for year in {$year_start..${year_end}}; do
	if [[ $year -le 2005 ]]; then run='historical'; else run='rcp85'; fi
	wdir_anom="/home/portal/work_big/CORDEX/${run}/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/anom/"
	file_year_tmp="${wdir_anom}tmp_anom${year}.nc"
	file_anom="${wdir_anom}${varname}-anom_EUR-11_${gcm_name}_${run}_${memb_name}_${regm_name}_day_${year}_remapbil-to-05res.nc"
	mkdir -p $wdir_anom
	cdo -O selyear,$year/$year $files $file_year_tmp
	cdo sub $file_year_tmp $file_clim_sm $file_anom

	rm $file_year_tmp
done

