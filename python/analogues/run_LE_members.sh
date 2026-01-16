#!/bin/bash

# Loop over kba..kbz
for memb in kb{a..z}; do
    echo "Running for member $memb"
    python3 /home/alice/Desktop/work/git/myISACcode/python/analogues/find_LE-analogue-dates_mslp-z500.py "$memb"
done

# Loop over kca..kcx
for memb in kc{a..x}; do
    echo "Running for member $memb"
    python3 /home/alice/Desktop/work/git/myISACcode/python/analogues/find_LE-analogue-dates_mslp-z500.py "$memb"
done

