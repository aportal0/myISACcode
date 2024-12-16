#!/bin/sh

#SBATCH -J HourlyToDaily
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --output=hourlyToDaily_%j_out.log
#SBATCH --error=hourlyToDaily_%j_err.log 
#SBATCH --partition=batch

field=zg
nbatch=5
expm='historical'
iens=1
ii=1

ddir=/home/portal/work_big/CRCM5-LE
ensm_list=`ls -d ${ddir}/${field}/k* | cut -d / -f7`
for ensm in ${ensm_list[@]} ; do
    
    ./hourlyTodaily_Zappa.sh $field $ensm $expm &

    ii=$(($ii+1))
    iens=$(($iens+1))
    
    if [[ $ii -gt $nbatch ]]; then
     	ii=1
     	wait
    fi   
done

exit 0
