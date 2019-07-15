#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A ebudur
#SBATCH -J m1_t1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 10
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model1_task2/slurm_outputs/
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=04:15:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model1_task2/slurm_outputs/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model1_task2/slurm_outputs/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################


python /truba/home/ebudur/tse-s2v/utils/punctuation_stripper.py
