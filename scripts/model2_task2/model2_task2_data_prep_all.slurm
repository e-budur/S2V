#!/bin/bash
#SBATCH -p barbun
#SBATCH -A ebudur
#SBATCH -J m2_t2_p
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model2_task2/slurm_outputs/data_prep/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=23:59:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model2_task2/slurm_outputs/data_prep/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model2_task2/slurm_outputs/data_prep/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################

NUM_WORDS=100001
DATA_FOLDER="/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR"
OUTPUT_DIR="/truba/home/ebudur/tse-s2v/data/bulk_sentences/tr/TR-SP-TFRecords"
TOKENIZED_FILES="$DATA_FOLDER/*/*.no_punct.txt"
SENTENCEPIECE_MODEL="$DATA_FOLDER/sentencepiece/spm.model"

python /truba/home/ebudur/tse-s2v/src/data/preprocess_dataset.py --input_files "$TOKENIZED_FILES" --output_dir $OUTPUT_DIR --num_words $NUM_WORDS --max_sentence_length 50 --case_sensitive False --sentencepiece_model=$SENTENCEPIECE_MODEL
