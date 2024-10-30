#!/bin/bash
name_file='tas_EUR-11_CCCma-CanESM2_historical_r1-r1i1p1_OURANOS-CRCM5_kba'

cd kba
# loop years
for y in {1955..1955}
do
	cd /work_big/users/portal/CRCM5-LE/tas/kba/$y
	echo `ls`
	# loop months
	for m in $(seq -w 1 12)
	do
		# compute daily mean
		# cdo daymean ${name_file}_3h_${y}${m}.nc ${name_file}_daily_${y}${m}.nc 
		cdo timmean ${name_file}_daily_${y}${m}.nc ${name_file}_daily_${y}${m}_new.nc
	        rm ${name_file}_daily_${y}${m}.nc 
	        mv ${name_file}_daily_${y}${m}_new.nc ${name_file}_daily_${y}${m}.nc 
	done	
	cd ..
done
