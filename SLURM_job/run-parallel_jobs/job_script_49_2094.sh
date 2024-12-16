#!/bin/bash
#SBATCH --job-name=dmean_psl_49_2094        # Job name
#SBATCH --mem=3000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_49_2094.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_49_2094.err    # Error file
#SBATCH --partition=batch

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 49 2094 3 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 49 2094 3
