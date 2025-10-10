#!/bin/bash

YEAR_START=2099
YEAR_END=2099

# Members
memb=$1

#SBATCH -J ${memb}_pr_daily
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --output=${memb}.out
#SBATCH --error=${memb}.out
#SBATCH --partition=batch


# Pre- and post-processing
nckspre="ncks -O -L 0 --chunk_policy g3d --cnk_dmn plev,1 --cnk_dmn rlon,50 --cnk_dmn rlat,50 --cnk_dmn time,1 "
nckspost="ncks -O --chunk_policy g3d --cnk_dmn time,31 "


# Loop over year, mon
for YEAR in $(seq $YEAR_START $YEAR_END); do
	cd "/home/portal/work_big/CRCM5-LE/pr/${YEAR}"
	memb_code=`ls pr_EUR-11_CCCma-CanESM2_rcp85_*_OURANOS-CRCM5_${memb}_1h_${YEAR}01.nc | grep -oP "(?<=rcp85_)[^_]+(?=_OURANOS)"`
	# Create tmp dir
	mkdir -p tmp_${memb}
	
	echo "Processing YEAR=$YEAR MEMBER=$memb"
	for mon in $(seq -w 1 12); do
		
		# Input and output file names (adapt to your naming convention)
      		infile="pr_EUR-11_CCCma-CanESM2_rcp85_${memb_code}_OURANOS-CRCM5_${memb}_1h_${YEAR}${mon}.nc"
      		outfile="pr_${memb}_daily_${YEAR}${mon}.nc"
		# Pre-processing
		$nckspre "$infile" "tmp_${memb}/${infile}tmp"
		# Daymean
		cdo daysum "tmp_${memb}/${infile}tmp" "tmp_${memb}/${outfile}tmp" # not daysum, but consistent with Zappa and Marra
	done
	
	# Merge monthly files
	yearfile="pr_daysum_rcp85_${YEAR}01-${YEAR}12_${memb}.nc"
	cdo mergetime "tmp_${memb}/pr_${memb}_daily_${YEAR}*" "tmp_${memb}/${yearfile}tmp"
	$nckspost "tmp_${memb}/${yearfile}tmp" "${yearfile}"
	
	# Remove tmp directory
	rm -r tmp_${memb}

done



