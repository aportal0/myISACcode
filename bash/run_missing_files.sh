#!/bin/bash

# Define members and their corresponding years
members=(34 35 40 44 45 45 46 47)
years=(2087 1962 1961 2034 1968 2031 2096 2058)

# Loop through each member–year pair
for i in "${!members[@]}"; do
  member="${members[$i]}"
  year="${years[$i]}"
  echo "Running: python3 compute_daily-mean_mpc.py zg $member $year 2"
  python3 /home/portal/script/python/compute_daily-mean_mpc.py zg "$member" "$year" 2
  if [ $? -ne 0 ]; then
    echo "Error running for $member $year" >&2
    exit 1
  fi
done
