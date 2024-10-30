#!/bin/bash
#SBATCH --job-name=dmean_psl_6_2013   # Job name
#SBATCH --ntasks=1                                      # One job (Python script) but multiple threads 
#SBATCH --cpus-per-task=4                 # Request no. CPUs
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_6_2013.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_6_2013.err    # Error file

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
time srun python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 6 2013 4 ## FASTEST OPTION
# time srun python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 6 2013 4
