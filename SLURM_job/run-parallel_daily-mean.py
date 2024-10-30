import os
import functions_slurm as fs

# Input variables
varname = 'psl'                 # Choice between 'tas', 'psl', 'zg', ...
year_range = [1983,1983]
memb_range = [18,18]
ncpus_per_job = 4
nbatch = 1
username = 'portal'

# Loop over members and years
for memb in range(memb_range[0], memb_range[1]+1):
    for year in range(year_range[0], year_range[1]+1):

        # Define the SLURM job script content
        job_script = f"""#!/bin/bash
#SBATCH --job-name=dmean_{varname}_{memb}_{year}        # Job name
#SBATCH --ntasks=1                                      # One job (Python script) but multiple threads 
#SBATCH --cpus-per-task={ncpus_per_job}                 # Request no. CPUs
#SBATCH --mem=3000
#SBATCH --output=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_{year}.out  # Output file
#SBATCH --error=/home/portal/script/SLURM_job/run-parallel_jobs/job_{memb}_{year}.err    # Error file

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