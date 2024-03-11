#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -o interpretability_1131.out
eval "$(conda shell.bash hook)"

conda activate deepattpan
echo 'Job starts!'
python /home/ruofan/bcr/split/interpretable/1131/interpre.py
echo 'Job ends!'
