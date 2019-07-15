#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A ebudur
#SBATCH -J m1_t1_eval_all_multinli_matched
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 10
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/eval/
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=15-00:00:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/eval/slurm-multinli-matched-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/eval/slurm-multinli-matched-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################

WORK_DIR="/truba/home/ebudur/tse-s2v" #needs to be changed according to your specific environment

TASK='MULTINLI-MATCHED'

MDLS_PATH="/truba_scratch/ebudur/data/results"
MDL_CFGS="$WORK_DIR/model_configs"
#GLOVE_PATH="$WORK_DIR/data/word_embeddings/glove"

CFG="UMBC"

DATA="$WORK_DIR/data/sem_sim/eng/MultiNLI_1.0/"

python /truba/home/ebudur/tse-s2v/src/evaluate.py --eval_task=$TASK --data_dir=$DATA --model_config="$MDL_CFGS/$CFG/eval.json" --results_path="$MDLS_PATH"

