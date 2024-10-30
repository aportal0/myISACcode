#!/bin/bash
#SBATCH --job-name=dmean_psl_18_1971        # Job name
#SBATCH --ntasks=1                                      # One job (Python script) but multiple threads 
#SBATCH --cpus-per-task=2                 # Request no. CPUs
#SBATCH --mem=2000
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_18_1971.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_18_1971.err    # Error file

# Activate conda environment
source /home/${USER}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time srun python3 /home/${USER}/script/python/compute_daily-mean_mpc.py psl 18 1971 2 ## FASTEST OPTION
time python3 /home/${USER}/script/python/compute_daily-mean_dask.py psl 18 1971 2
