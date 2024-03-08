#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o interpretability_645.out
eval "$(conda shell.bash hook)"

conda activate deepattpan
echo 'Job starts!'
python /home/ruofan/bcr/split/interpretable/645/interpre.py
echo 'Job ends!'
