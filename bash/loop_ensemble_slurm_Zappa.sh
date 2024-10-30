#!/bin/sh

#SBATCH -J HourlyToDaily
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --output=hourlyToDaily_%j_out.log
#SBATCH --error=hourlyToDaily_%j_err.log 
#SBATCH --partition=batch

field=pr
nbatch=5
expm='rcp85'
iens=1
ii=1

ensm_list=`ls -d ${field}/k* | cut -d / -f2`
for ensm in ${ensm_list[@]} ; do
    
    ./hourlyTodaily.sh $field $ensm $expm &

    ii=$(($ii+1))
    iens=$(($iens+1))
    
    if [[ $ii -gt $nbatch ]]; then
     	ii=1
     	wait
    fi   
done

exit 0
