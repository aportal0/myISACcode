#!/bin/bash

# Variable
namevar='z500'
# Input and output directories
INPUT_DIR="/work_big/users/clima/portal/ERA5/${namevar}/"
OUTPUT_DIR="/work_big/users/clima/portal/ERA5/${namevar}/"
mkdir -p "$OUTPUT_DIR"  # Ensure output directory exists

# Loop through years
for year in {2024..2024}; do
#    if [[ $namevar == 'z500' ]]; then 
#	    input_file0="${INPUT_DIR}ERA5_${namevar}_NH_6hr_${year}.grib"
#    fi
    input_file="${INPUT_DIR}ERA5_${namevar}_NH_6hr_${year}.nc"
    output_file="${OUTPUT_DIR}ERA5_${namevar}_NH_daily_${year}.nc"
   
    # Process z500
    if [[ $namevar == 'z500' || $namevar == 'mslp' ]]; then
	    cdo daymean -del29feb "$input_file" "$output_file"
	    rm -f $input_file0
    elif [ $namevar == 'mslp' ]; then
	    cdo daymean -del29feb -sellonlatbox,0,360,0,90 "$input_file" "$output_file"
    fi
     
    echo "Processed: $year -> $output_file"
done

