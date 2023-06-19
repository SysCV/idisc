#!/bin/bash

#SBATCH --job-name=iDisc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --tmp=200G

DATA=${1}
CFG=${2}
BP=${3}

MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
IDISC_REPO="$HOME/Workspace/idisc" #change as needed!!

source ~/setting
source ${HOME}/torch110/bin/activate

echo "Store datasets at ${BP}/datasets"
mkdir -p ${BP}/datasets
for datasingle in ${DATA}; do
    echo "Unzip $datasingle dataset"
    tar -xf ${datasingle} -C ${BP}/datasets
done

echo "Start script"
cd ${IDISC_REPO} && srun --cpus-per-task=4 --gpus=4 --gres=gpumem:20G python -u ${IDISC_REPO}/idisc/scripts/train.py --config-file ${CFG} --base-path ${BP} --master-port ${MASTER_PORT} --distributed
