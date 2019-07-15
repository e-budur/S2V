#!/bin/bash
#SBATCH -p barbun
#SBATCH -A ebudur
#SBATCH -J m1_t1_p
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 2
#SBATCH --workdir=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/data_prep/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emrahbudur@gmail.com
#SBATCH --time=23:15:00
#SBATCH --output=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/data_prep/slurm-%j.out
#SBATCH --error=/truba/home/ebudur/tse-s2v/scripts/model1_task1/slurm_outputs/data_prep/slurm-%j.err

###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################

NUM_WORDS=100001
OUTPUT_DIR="/truba/home/ebudur/tse-s2v/data/bulk_sentences/en/UMBC-SMALL-TFRecords"
#VOCAB_FILE="../../data/word_embeddings/glove/glove.840B.300d_dictionary.txt"  
TOKENIZED_FILES="/truba/home/ebudur/tse-s2v/data/bulk_sentences/en/UMBC-SMALL/*.txt"

python /truba/home/ebudur/tse-s2v/src/data/preprocess_dataset.py --input_files "$TOKENIZED_FILES" --output_dir $OUTPUT_DIR --num_words $NUM_WORDS --max_sentence_length 50 --case_sensitive False
