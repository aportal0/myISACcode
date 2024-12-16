#!/bin/bash

varname=$1
ensname=$2
year_start=$3
year_end=$4

#SBATCH -J ${ensname}_clim
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${ensname}.out
#SBATCH --error=${ensname}.out
#SBATCH --partition=batch

# Parameters
no_years=$(( ${year_end} - ${year_start} + 1 )) 
wdir="/home/portal/work_big/CRCM5-LE/${varname}/${ensname}/"
scriptdir="/home/portal/script/bash/"
window_rmean=31

# Code ensemble
tmp_file=`ls ${wdir}/1955/${varname}*195501.nc`
enscode=`echo $tmp_file | rev | cut -d _ -f 5 | rev`
echo "$enscode"

#-----------------
# Remap to 0.5 degree resolution and compute daily climatology
for mon in {01..12}; do
	mkdir -p "${wdir}clim/"
	out_clim="${wdir}clim/${varname}_EUR-11_CCCma-CanESM2_${enscode}_OURANOS-CRCM5_${ensname}_daily_${mon}_clim${year_start}-${year_end}_05res.nc"
        for (( y=${year_start}; y<=${year_end}; y++ )); do
                mkdir -p ${wdir}$y/res05
        	
                # Define name files
		if [[ $y -le 2005 ]]; then name_run='historical'; else name_run='rcp85'; fi
        	in_f="${wdir}$y/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}.nc"
        	out_f="${wdir}$y/res05/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}_remapbil-to-05res.nc"
		echo "$in_f"
        	
                # Regrid mon,year file
		ncks -A ${scriptdir}rotated_pole.nc $in_f
        	cdo remapbil,${scriptdir}grid_05res.txt ${in_f} ${out_f}
		
                # Accumulate clim
                if [ ! -f "$out_clim" ]; then
			# If the accumulated file doesn't exist, initialize it with the first file
			cp "$out_f" "$out_clim"
		else
			# Accumulate data by adding the current file to the accumulated file
			cdo -O add "$out_clim" "$out_f" "$out_clim.tmp"
                        mv "$out_clim.tmp" "$out_clim"
		fi
        done
        # Divide by # years
       	cdo divc,$no_years "$out_clim" "$out_clim.tmp"
        mv "$out_clim.tmp" "$out_clim"
done

#-----------------
# Compute running mean of daily climatology using a window $window_rmean days
concat_clim="${wdir}clim/concatenated_clim.nc" 
ext_clim="${wdir}clim/extended_clim.nc" 
ext_clim_sm="${wdir}clim/extended_clim_sm${window_rmean}d.nc"

# Merge and extend daily climatology
cdo mergetime "${wdir}clim/${varname}_EUR-11_CCCma-CanESM2_${enscode}_OURANOS-CRCM5_${ensname}_daily_"{01..12}"_clim${year_start}-${year_end}_05res.nc" "$concat_clim"
cdo cat -selmon,12 "$concat_clim" "$concat_clim" -selmon,1 "$concat_clim" "$ext_clim"
cdo runmean,$window_rmean "$ext_clim" "$ext_clim_sm"

# Split by month and save
month_lengths=(31 28 31 30 31 30 31 31 30 31 30 31)
start_day=17
for mon in {01..12}; do
	# Name output file
	out_clim_sm="${wdir}clim/${varname}_EUR-11_CCCma-CanESM2_${enscode}_OURANOS-CRCM5_${ensname}_daily_${mon}_clim${year_start}-${year_end}_sm${window_rmean}d_05res.nc" 
        # Select days in month
	days_in_month=${month_lengths[$((10#${mon}-1))]}
        end_day=$((start_day + days_in_month - 1))
	echo "$days_in_month" "$start_day" "$end_day"
	cdo seltimestep,$start_day/$end_day "$ext_clim_sm" "$out_clim_sm"
	start_day=$((end_day + 1))
done
rm "$concat_clim" "$ext_clim" "$ext_clim_sm"

#-----------------
# Compute anomaly from smoothed climatology
for mon in {01..12}; do
	clim_f="${wdir}clim/${varname}_EUR-11_CCCma-CanESM2_${enscode}_OURANOS-CRCM5_${ensname}_daily_${mon}_clim${year_start}-${year_end}_sm${window_rmean}d_05res.nc" 
        for (( y=${year_start}; y<=${year_end}; y++ )); do
		# Define name files
		if [[ $y -le 2005 ]]; then name_run='historical'; else name_run='rcp85'; fi
        	full_f="${wdir}$y/res05/${varname}_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}_remapbil-to-05res.nc"
        	anom_f="${wdir}$y/res05/${varname}-anom_EUR-11_CCCma-CanESM2_${name_run}_${enscode}_OURANOS-CRCM5_${ensname}_daily_${y}${mon}_remapbil-to-05res.nc"
		cdo sub "$full_f" "$clim_f" "$anom_f"
	done
done
