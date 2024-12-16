#!/bin/bash
#SBATCH --job-name=dmean_psl_24_2098        # Job name
#SBATCH --cpus-per-task=4                 # Request no. CPUs
#SBATCH --mem=3000
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_24_2098.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_24_2098.err    # Error file

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 24 2098 4 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 24 2098 4
