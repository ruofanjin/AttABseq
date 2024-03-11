#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH -o interpretability_1101.out
eval "$(conda shell.bash hook)"

conda activate deepattpan
echo 'Job starts!'
python /home/ruofan/bcr/split/interpretable/1101/interpre.py
echo 'Job ends!'
