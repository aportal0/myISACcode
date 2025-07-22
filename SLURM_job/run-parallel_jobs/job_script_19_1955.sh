#!/bin/bash
#SBATCH --job-name=dmean_zg_19_1955        # Job name
#SBATCH --mem=2000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_19_1955.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_19_1955.err    # Error file
#SBATCH --partition=batch

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py zg 19 1955 3 ## FASTEST OPTION
# time python3 /home/${USER}/script/python/compute_daily-mean_dask.py zg 19 1955 3
