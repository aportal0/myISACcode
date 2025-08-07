#!/bin/bash

varname="pr"
ensname=$1

#SBATCH -J ${ensname}_clim
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${ensname}.out
#SBATCH --error=${ensname}.out
#SBATCH --partition=batch

# Parameters
no_years=$(( ${year_end} - ${year_start} + 1 )) 
wdir="/home/portal/work_big/CRCM5-LE/${varname}/"
scriptdir="/home/portal/script/bash/"
datadir="/mnt/naszappa/CRCM5-LE/CanESM2_driven/pr/daily/"

# Code ensemble
tmp_file=`ls /home/portal/work_big/CRCM5-LE/psl/${ensname}/1955/psl*195501.nc`
enscode=`echo $tmp_file | rev | cut -d _ -f 5 | rev`
echo "$enscode"


#-----------------

# Remap to 0.5 degree resolution

name_run=('historical' 'rcp85')
range_years=('195501-200512' '200601-209812')

for i in "${!name_run[@]}"; do
  run="${name_run[$i]}"
  years="${range_years[$i]}"
  
  # Define names of input and output 
  in_f="${datadir}${varname}_daysum_${run}_${years}_${ensname}.nc"
  tmp_f="${wdir}${varname}_daysum_${run}_${years}_${ensname}.nc"
  out_f="${wdir}${varname}_daysum_${run}_${years}_${ensname}_remapbil-to-05res.nc"
  
  # Regrid input file
  cp $in_f $tmp_f
  ncks -A ${scriptdir}rotated_pole.nc ${tmp_f}
  cdo remapbil,${scriptdir}grid_05res.txt ${tmp_f} ${out_f}
  rm $tmp_f ${wdir}*tmp
done

