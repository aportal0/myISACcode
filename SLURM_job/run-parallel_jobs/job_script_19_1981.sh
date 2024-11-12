#!/bin/bash
#SBATCH --job-name=dmean_psl_19_1981        # Job name
#SBATCH --cpus-per-task=4                 # Request no. CPUs
#SBATCH --mem=3000
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_19_%a.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_19_%a.err    # Error file
#SBATCH --array=0-7

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# Define the year list as a bash array
YEAR_LIST=(1981 1982 1983 1984 1985 1986 1987 1988)
YEAR=${YEAR_LIST[$SLURM_ARRAY_TASK_ID]}

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 19 $YEAR 4 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 19 $YEAR 4
