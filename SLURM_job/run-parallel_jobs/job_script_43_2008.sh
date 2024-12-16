#!/bin/bash
#SBATCH --job-name=dmean_psl_43_2008        # Job name
#SBATCH --mem=3000
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_43_2008.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_43_2008.err    # Error file
#SBATCH --partition=batch

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 43 2008 3 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 43 2008 3
