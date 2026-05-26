#!/bin/bash

varname=$1
gcm_name=$2
regm_name=$3
memb_name=$4

#SBATCH -J remap_CORDEX
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${gcm_name}_${regm_name}_${memb_name}.out
#SBATCH --error=${gcm_name}_${regm_name}_${memb_name}.err
#SBATCH --partition=batch

# Parameters
scriptdir="/home/portal/work/myISACcode/bash/"

#-----------------

# Remap to 0.5 degree resolution

name_run=('historical' 'rcp85')

for i in "${!name_run[@]}"; do
	# Name run
  	run="${name_run[$i]}"
	# Define directories
	wdir="/home/portal/work_big/CORDEX/${run}/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
	datadir="/mnt/naszappa/CORDEX/output/CORDEX/${run}/${gcm_name}/${regm_name}/day/${memb_name}/${varname}/"
	mkdir -p $wdir
	# Loop over files in datadir
	for f_in in "${datadir}"*.nc; do
		tmp=${f_in##*/}
		tmp=${tmp%.nc}
		year_range=${tmp##*_}
		# Define name of output file
		f_out="${wdir}${tmp}_remapbil-to-05res.nc"
		f_tmp="${wdir}${tmp}.nc_tmp"
 
		# Regrid
		cp $f_in $f_tmp
		ncks -A ${scriptdir}rotated_pole_CORDEX.nc ${f_tmp}
                cdo remapbil,${scriptdir}grid_05res_CORDEX.txt ${f_tmp} ${f_out}
  		rm $f_tmp
	done
done
