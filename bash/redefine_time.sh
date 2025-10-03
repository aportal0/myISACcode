#!/bin/bash
# Redefine time axis in monthly NetCDF files using CDO

MEMB=kbv
CODE_MEMB=r3-r2i1p1
DATA_DIR=/work_big/users/clima/portal/CRCM5-LE/psl 
START_YEAR=2004 
END_YEAR=2023

for YEAR in $(seq $START_YEAR $END_YEAR); do
	YEAR_DIR="${DATA_DIR}/${MEMB}/${YEAR}/res05"
	for MON in $(seq -w 1 12); do
		
		# Define filename
		if [[ $YEAR -lt 2006 ]]; then
			STYPE="historical"
		else
			STYPE="rcp85"
		fi
		fpath="${YEAR_DIR}/psl-anom_EUR-11_CCCma-CanESM2_${STYPE}_${CODE_MEMB}_OURANOS-CRCM5_${MEMB}_daily_${YEAR}${MON}_remapbil-to-05res.nc"
		
		# Define start date of this month
    		START_DATE="${YEAR}-${MON}-01"
                # Redefine time axis as daily steps within the month
    		cdo settaxis,$START_DATE,00:00:00,1day "$fpath" "${YEAR_DIR}/tmp${MON}.nc"
		rm $fpath
		mv ${YEAR_DIR}/tmp${MON}.nc $fpath
	done
done

