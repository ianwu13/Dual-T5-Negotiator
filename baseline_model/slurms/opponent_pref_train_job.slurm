#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:15:00

# Change to proper directory
if [[ "$PWD" =~ "negotiation-dialogue/baseline_model/slurms" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "negotiation-dialogue/baseline_model/slurms" FOR THIS PROJECT'
    return -1
fi

# load modules
module load python/3.7.6

# load libraries
pip install -r requirements.txt

# run
python3 scripts/train_opponent_pref.py