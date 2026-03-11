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
	
	# Compute monthly mean of box average 
    	for mon in {01..12}; do
        	in_f="${wdir}$y/res05/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}_remapbil-to-05res.nc"
		tmp_f=${wdir}$y/$y${mon}_tmp.nc
                
		# Regrid mon,year file
		if [ ! -e "$in_f" ]; then
			mkdir -p $wdir$y/res05/
			in_f0="${wdir}$y/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}.nc"
			ncks -A ${scriptdir}rotated_pole.nc $in_f0
        		cdo remapbil,${scriptdir}grid_05res.txt ${in_f0} ${in_f}
		fi
		# Compute mean
    		cdo monmean -fldmean -sellonlatbox,-10,40,30,45 ${in_f} ${wdir}$y/res05/$y${mon}_tmp.nc
	done
	
	# Merge monthly files
	out_f="${wdir}$y/res05/${varname}_mean-Medregion_${ensname}_monthly_${y}.nc"
	cdo mergetime ${wdir}$y/res05/*tmp.nc ${out_f}
	rm ${wdir}$y/res05/*tmp* ${wdir}$y/*tmp* 
done

list_out=(${wdir}*/res05/*mean-Medregion*)
monthly_allyears="${wdir}${varname}_mean-Medregion_${ensname}_monthly_${year_start}-${year_end}.nc"
yearly_allyears="${wdir}${varname}_mean-Medregion_${ensname}_yearly_${year_start}-${year_end}.nc"
 
if [ "${#list_out[@]}" -eq 145 ]; then
    cdo mergetime "${list_out[@]}" $monthly_allyears
    cdo yearmean $monthly_allyears $yearly_allyears
else
    echo "Error in member ${ensname}: expected 145 files, found ${#files[@]}"
fi
