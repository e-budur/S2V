#!/bin/bash
#SBATCH -p barbun-cuda
#SBATCH -A ebudur
#SBATCH -J m1_t1_all
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 10
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=10-00:00:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################


WORK_DIR="/truba/home/ebudur/tse-s2v" #needs to be changed according to your specific environment
RESULTS_HOME="/truba_scratch/ebudur/data/results"
MDL_CFGS="$WORK_DIR/model_configs"
#GLOVE_PATH="$WORK_DIR/data/word_embeddings/glove"
          
DATA_DIR="/truba_scratch/ebudur/data/bulk_sentences/en/UMBC-TFRecords"
NUM_INST=133989387 # Number of sentences

CFG="UMBC"
      
BS=400
SEQ_LEN=30
    
python /truba/home/ebudur/tse-s2v/src/train.py --results_path="$RESULTS_HOME/$CFG" --input_file_pattern="$DATA_DIR/train-?????-of-00100" --train_dir="$RESULTS_HOME/$CFG/train" --learning_rate_decay_factor=0 --batch_size=$BS --sequence_length=$SEQ_LEN --nepochs=1 --num_train_inst=$NUM_INST --save_model_secs=1800 --Glove_path=$GLOVE_PATH --model_config="$MDL_CFGS/$CFG/train.json"




