#!/bin/bash

# --- Configuration ---
SRC_UUID="1bc1deb2-6fcf-11ec-b2c3-1b99bfd4976a"
DST_UUID="cdf48be0-7a79-11ef-91bf-f95c6daa0003"
SRC_BASE="/ClimEx/ClimEx_shared_links/CanESM2_driven_50_members/pr"
DST_BASE="/work_big/users/clima/portal/CRCM5-LE/pr/2099"
LABEL="BatchTransfer_precip"

# --- Function to run a transfer ---
run_transfer() {
    local subdir=$1
    echo "Starting transfer for $subdir ..."
    globus transfer \
        --recursive \
        --label "${LABEL}_${subdir}" \
        "${SRC_UUID}:${SRC_BASE}/${subdir}/2099" "${DST_UUID}:${DST_BASE}"
}

# # --- Generate kba..kbz ---
# for letter in {a..z}; do
#     subdir="kb${letter}"
#     run_transfer "$subdir"
#     echo "Waiting 60 seconds..."
#     sleep 60
# done

# --- Generate kca..kcx ---
for letter in {q..x}; do
    subdir="kc${letter}"
    run_transfer "$subdir"
    echo "Waiting 60 seconds..."
    sleep 60
done

echo "All transfers submitted."

