#!/bin/sh

# Parameters for job
field=zg     # slp, zg, tas
start_y=2080
end_y=2099

# Parameter and function for limiting # running jobs
MAX_RUNNING_JOBS=10
get_running_jobs_count() {
	# Replace `username` with your actual username if necessary
	squeue -u portal --noheader | grep -c ' R '
}

# Run job by ensemble member
ddir=/home/portal/work_big/CRCM5-LE/
ensm_list=(`ls -d ${ddir}/${field}/k* | cut -d / -f8`)
for ensm in ${ensm_list[@]} ; do
        # Submit while checking for maximum number of jobs
	while true; do
		n_jobs=$(get_running_jobs_count)
		echo "member $ensm running jobs $n_jobs MAX_jobs $MAX_RUNNING_JOBS"
		if (( n_jobs < MAX_RUNNING_JOBS )); then
			echo "Submitting job $ensm"
			sbatch /home/portal/script/bash/script_remap_clim.sh $field $ensm $start_y $end_y &
			sleep 5
			break
		fi   

		echo "Waiting until running jobs less than MAX_jobs (5)"
		sleep 100
	done
done

exit 0

