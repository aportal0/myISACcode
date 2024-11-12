import os
import functions_slurm as fs
import numpy as np

# Input variables
varname = 'psl'                 # Choice between 'tas', 'psl', 'zg', ...
year_range = [1981,1988]
memb_range = [19,19]
ncpus_per_job = 4
nbatch = 8
nbatch_submit = 1
mem_per_job = 3000
username = 'portal'

# Loop over members and years
for memb in range(memb_range[0], memb_range[1]+1):
    for year in range(year_range[0], year_range[1]+1, nbatch):
        yearN = year + nbatch-1 if year + nbatch <= year_range[1] else year_range[1]
        year_list = np.arange(year,yearN+1)
        # Define the SLURM job script content
        job_script = f"""#!/bin/bash
#SBATCH --job-name=dmean_{varname}_{memb}_{year}        # Job name
#SBATCH --cpus-per-task={ncpus_per_job}                 # Request no. CPUs
#SBATCH --mem={mem_per_job}
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_%a.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_%a.err    # Error file
#SBATCH --array=0-{nbatch-1}

# Activate conda environment
source /home/${{USER}}/.bashrc
source activate myenv

# Define the year list as a bash array
YEAR_LIST=({" ".join(map(str, year_list))})
YEAR=${{YEAR_LIST[$SLURM_ARRAY_TASK_ID]}}

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${{USER}}/script/python/compute_daily-mean_mpc.py {varname} {memb} $YEAR {ncpus_per_job} ## FASTEST OPTION
time python3 /home/${{USER}}/script/python/compute_daily-mean_dask.py {varname} {memb} $YEAR {ncpus_per_job}
"""

        # Save the job file and make executable
        job_file_path = f'/home/portal/script/SLURM_job/run-parallel_jobs/job_script_{memb}_{year}.sh'
        with open(job_file_path, 'w') as f:
            f.write(job_script)
        os.chmod(job_file_path, 0o755)
        

        # If below limit batch maximum (nbatch), submit the job
        fs.limit_batch_submit(nbatch_submit+1, username)
        os.system(f'sbatch {job_file_path}')
        print(f'Job submitted for member {memb} and year {year}')
