#!/bin/bash

var=$1
ensname=$2
year_start=$3
year_end=$4
if [ "$var" = "zg" ]; then varname="zg500"; else varname="$var"; fi

#SBATCH -J ${ensname}_clim
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${ensname}.out
#SBATCH --error=${ensname}.out
#SBATCH --partition=batch

# Parameters
no_years=$(( ${year_end} - ${year_start} + 1 )) 
wdir="/home/portal/work_big/CRCM5-LE/${var}/${ensname}/"
scriptdir="/home/portal/work/myISACcode/bash/"

# Code ensemble
tmp_file=`ls ${wdir}/1955/${varname}*195501.nc`
enscode=`echo $tmp_file | rev | cut -d _ -f 5 | rev`
echo "$enscode"

#-----------------
# Loop over years
for (( y=${year_start}; y<=${year_end}; y++ )); do
	# Define name run
	if [[ $y -le 2005 ]]; then name_run='historical'; else name_run='rcp85'; fi
	
	# Compute yearly average of mean in lon-lat box
        in_f="${wdir}$y/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}*.nc"
	out_f="${wdir}$y/${varname}_${ensname}_annual_${y}_Mediterranean-region.nc"
	cdo yearmean -fldmean -mergetime -sellonlatbox,-10,40,30,45 ${in_f} ${out_f}
done
