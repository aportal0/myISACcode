#!/bin/bash

# Parameters for job
field="zg500"
list_gcmod=("MPI-M-MPI-ESM-LR" "ICHEC-EC-EARTH")
list_regmod=("COSMO-crCLIM-v1-1" "RCA4")
list_memb=("r1i1p1" "r2i1p1" "r3i1p1")  
# Year range for clim
year0=2070
yearN=2099

# Parameter and function for limiting # running jobs
MAX_RUNNING_JOBS=3
get_running_jobs_count() {
	# Replace `username` with your actual username if necessary
	squeue -u portal --noheader | grep -c ' R '
}

# Run job by ensemble member
ddir=/home/portal/work_big/
for gcmod in ${list_gcmod[@]} ; do
	for regmod in ${list_regmod[@]} ; do
		for memb in ${list_memb[@]} ; do

			# Correct name member
			if [[ "$gcmod" == "ICHEC-EC-EARTH" && "$memb" == "r2i1p1" ]]; then
				memb="r12i1p1"
			fi
        		
			# Submit while checking for maximum number of jobs
			while true; do
				n_jobs=$(get_running_jobs_count)
				if (( n_jobs < MAX_RUNNING_JOBS )); then
					echo "Submitting job"
					echo "gcm:$gcmod, regm:$regmod, memb:$memb running jobs $n_jobs MAX_jobs $MAX_RUNNING_JOBS"
					sbatch /home/portal/work/myISACcode/bash/script_clim_CORDEX.sh $field $gcmod $regmod $memb $year0 $yearN &
					sleep 5
					break
				fi
			done	
			echo "Waiting until running jobs ($n_jobs) less than MAX_jobs ($MAX_RUNNING_JOBS)"
			sleep 20
		done
	done
done

exit 0
