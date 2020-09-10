#!/bin/bash -l
#
#BATCH --mail-type=ALL
#SBATCH --mail-user Ben.Zhang@nyumlangone.org
#SBATCH --nodes=1
#SBATCH --mem=4GB
#SBATCH --partition=gpu8_short
#SBATCH --gres=gpu:1
#SBATCH --job-name=slurm_%j 
#SBATCH --output=train_rot2_%j.out
#SBATCH --error=train_rot2_%j.err
#SBATCH --time=00:02:00
#SBATCH --ntasks=1

module load anaconda3/gpu/5.2.0
conda activate /gpfs/scratch/bz957/pytorch1.2 

module unload anaconda3/gpu/5.2.0
module load cuda91/toolkit/9.1.85 
cd /gpfs/scratch/bz957/Motion_Correction/Pytorch-UNet-master

python -u train.py -i /gpfs/scratch/bz957/Motion_Correction/dataset/ds000101and2 \
                   -t /gpfs/scratch/bz957/Motion_Correction/dataset/rotation2 \
                   -e 1 \
                   -b 5
