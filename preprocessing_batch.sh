#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=18G
#SBATCH -J output/glutamatergic_batch
#SBATCH -o output/glutamatergic_batch-%j.out
#SBATCH -e output/glutamatergic_batch-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=avery_maytin@brown.edu

source ~/myenv/bin/activate

python preprocessing_batch.py