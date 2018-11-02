#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A ebudur
#SBATCH -J m1_t1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 10
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
##SBATCH --time=03:44:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba/home/ebudur/cuda-9.0/bin:/truba/home/ebudur/anaconda2/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba/home/ebudur/cuda-9.0/lib64
######################################################################
source activate tse-gpu
bash /truba/home/ebudur/tse-s2v/scripts/model1_task1/run-gpu.sh 
