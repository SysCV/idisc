#!/bin/bash

DATA=${1}
CFG=${2}
BP=${3}

MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
IDISC_REPO="$HOME/Workspace/idisc" #change as needed

echo "Store datasets at ${BP}/datasets"
mkdir -p ${BP}/datasets
for datasingle in ${DATA}; do
    echo "Unzip $datasingle dataset"
    tar -xf ${datasingle} -C ${BP}/datasets
done

echo "Start script"
cd ${IDISC_REPO} && python -u ${IDISC_REPO}/scripts/train.py --config-file ${CFG} --base-path ${BP} --master-port ${MASTER_PORT} --distributed
