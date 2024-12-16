#!/bin/bash
#SBATCH --job-name=dmean_psl_27_2073        # Job name
#SBATCH --cpus-per-task=4                 # Request no. CPUs
#SBATCH --mem=3000
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_27_2073.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_27_2073.err    # Error file

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 27 2073 4 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 27 2073 4