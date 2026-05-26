#!/bin/bash

varname="psl"
gcm_name="MPI-M-MPI-ESM-LR"
regm_name="COSMO-crCLIM-v1-1" 
memb_name="r1i1p1"
year_start=1950
year_end=1979

#SBATCH -J remap_CORDEX
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${gcm_name}_${regm_name}_${memb_name}.out
#SBATCH --error=${gcm_name}_${regm_name}_${memb_name}.err
#SBATCH --partition=batch

# Parameters
window_rmean=31

#-----------------

# Compute climatology and anomalies over range years

# Define file paths
wdir="/home/portal/work_big/CORDEX/*/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
files="${wdir}${varname}_EUR-11_${gcm_name}_*_${memb_name}_*${regm_name}_v1_day_*_remapbil-to-05res.nc"

# Define tmp and clim file names
wdir_clim="/home/portal/work_big/CORDEX/clim/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
file_merge="${wdir_clim}tmp_allfiles.nc"
file_clim="${wdir_clim}${varname}_EUR-11_${gcm_name}_${memb_name}_${regm_name}_v1_day_clim${year_start}-${year_end}_remapbil-to-05res.nc"
file_clim_sm="${wdir_clim}${varname}_EUR-11_${gcm_name}_${memb_name}_${regm_name}_v1_day_clim${year_start}-${year_end}_sm${window_rmean}d_remapbil-to-05res.nc"
file_tmp="${wdir_clim}tmp_clim${year_start}-${year_end}.nc"


# Compute clim and smooth with running mean
mkdir -p $wdir_clim
 
selected=()
for f in $files; do
	[[ $f =~ _([0-9]{4})[0-9]{4}-([0-9]{4})[0-9]{4}_ ]] || continue
	y1=${BASH_REMATCH[1]}
	y2=${BASH_REMATCH[2]}
    	# overlap test
    	(( $y2 >= $year_start && $y1 <= $year_end )) && selected+=("$f")
done

# Merge all relevant files
cdo -O mergetime "${selected[@]}" $file_merge

# Compute daily climatology
cdo -O del29feb -selyear,${year_start}/${year_end} $file_merge $file_tmp
cdo -O ydaymean $file_tmp $file_clim

# Compute running mean on climatology
cdo -O cat -selmon,12 "$file_clim" "$file_clim" -selmon,1 "$file_clim" "${file_clim}_ext"
cdo -O runmean,${window_rmean} "${file_clim}_ext" "${file_clim}_ext_sm"
cdo -O seltimestep,17/381 "${file_clim}_ext_sm" $file_clim_sm
rm ${wdir_clim}*_ext*

# Compute anomaly
for (( year=$year_start; year<=$year_end; year++ )); do
	if [[ $year -le 2005 ]]; then run='historical'; else run='rcp85'; fi
	wdir_anom="/home/portal/work_big/CORDEX/${run}/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/anom/"
	mkdir -p $wdir_anom
	file_year_tmp="${wdir_anom}tmp_anom${year}.nc"
	file_anom="${wdir_anom}${varname}-anom_EUR-11_${gcm_name}_${run}_${memb_name}_${regm_name}_day_${year}_remapbil-to-05res.nc"
	cdo -O selyear,$year $file_tmp $file_year_tmp
	cdo -O sub $file_year_tmp $file_clim_sm $file_anom

	rm $file_year_tmp
done
rm $file_tmp

