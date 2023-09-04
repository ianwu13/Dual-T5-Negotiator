#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:15:00

echo "TRAINING RESPONSE GENERATION MODEL"

# Change to proper directory
if [[ "$PWD" =~ "negotiation-dialogue/t5_x2_gated_attention/slurms" ]]
then 
    cd ..
else
    echo 'THIS SLURM JOB MUST BE RUN FROM "negotiation-dialogue/t5_x2_gated_attention/slurms"'
    return -1
fi

# load modules
module load python/3.7.6

# load libraries
pip install -r requirements.txt

# run
python3 scripts/pretrain_rg_model.py