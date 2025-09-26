import os
import functions_slurm as fs

# Input variables
varname = 'psl'                 # Choice between 'tas', 'psl', 'zg', ...
year_range = [2093,2093]
memb_range = [22,22]
ncpus_per_job = 3
nbatch = 10
mem_per_job = 3000
username = 'portal'

# Loop over members and years
for memb in range(memb_range[0], memb_range[1]+1):
    for year in range(year_range[0], year_range[1]+1):

        # Define the SLURM job script content
        job_script = f"""#!/bin/bash
#SBATCH --job-name=dmean_{varname}_{memb}_{year}        # Job name
#SBATCH --mem={mem_per_job}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncpus_per_job}
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_{year}.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_{year}.err    # Error file
#SBATCH --partition=batch

# Activate conda environment
source /home/${{USER}}/.bashrc
source activate myenv

# run python script
cd /home/portal/script/SLURM_job/
# time python3 /home/${{USER}}/script/python/compute_daily-mean_mpc.py {varname} {memb} {year} {ncpus_per_job} ## FASTEST OPTION
time python3 /home/${{USER}}/script/python/compute_daily-mean_dask.py {varname} {memb} {year} {ncpus_per_job}
"""

        # Save the job file and make executable
        job_file_path = f'/home/portal/script/SLURM_job/run-parallel_jobs/job_script_{memb}_{year}.sh'
        with open(job_file_path, 'w') as f:
            f.write(job_script)
        os.chmod(job_file_path, 0o755)
        

        # If below limit batch maximum (nbatch), submit the job
        fs.limit_batch_submit(nbatch, username)
        os.system(f'sbatch {job_file_path}')
        print(f'Job submitted for member {memb} and year {year}')
