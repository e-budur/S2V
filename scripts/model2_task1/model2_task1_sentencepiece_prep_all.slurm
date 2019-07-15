#!/bin/bash
#SBATCH -p barbun
#SBATCH -A ebudur
#SBATCH -J m2_t1_sp
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model2_task1/slurm_outputs/sentencepiece/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=23:59:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model2_task1/slurm_outputs/sentencepiece/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model2_task1/slurm_outputs/sentencepiece/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################

DATA_FOLDER="/truba/home/ebudur/tse-s2v/data/bulk_sentences/en/UMBC"
INPUT_FOLDER="$DATA_FOLDER/txt_tokenized/"
FILE_PATTERN="*.txt"
OUTPUT_MODEL_FILENAME="$DATA_FOLDER/sentencepiece/spm"
VOCAB_SIZE="8000"
INPUT_SENTENCE_SIZE="--input_sentence_size=10000000"

#find $DATA_FOLDER -type f -name $FILE_NAMES -exec cat {} + > $OUTPUT_FILE
rm $OUTPUT_MODEL_FILENAME*
python /truba/home/ebudur/tse-s2v/src/sp_train.py $INPUT_FOLDER $FILE_PATTERN $OUTPUT_MODEL_FILENAME $VOCAB_SIZE $INPUT_SENTENCE_SIZE

