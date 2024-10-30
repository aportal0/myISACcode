import os
import time

def get_running_jobs(user):
    """Get the number of running SLURM jobs for a specific user."""
    command = f"squeue -u {user} -h"
    output = os.popen(command).read().strip()
    jobs = output.splitlines()
    return len(jobs)

def limit_batch_submit(max_jobs, user):
    """Return when number of jobs is below the limit on the number of concurrent jobs."""
    # Check if the number of running jobs is below the limit
    while get_running_jobs(user) >= max_jobs:
        time.sleep(30)  # Wait 30 seconds before checking again
    return