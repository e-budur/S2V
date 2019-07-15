WORK_DIR="/truba/home/ebudur/tse-s2v" #needs to be changed according to your specific environment
RESULTS_HOME="$WORK_DIR/data/results"

OUTPUT_DIR="$WORK_DIR/data/bulk_sentences/en/UMBC-SMALL-TFRecords"

NUM_WORDS=100001
#VOCAB_FILE="../../data/word_embeddings/glove/glove.840B.300d_dictionary.txt"  
TOKENIZED_FILES="$WORK_DIR/data/bulk_sentences/en/UMBC-SMALL/*.txt"

python /truba/home/ebudur/tse-s2v/src/data/preprocess_dataset.py --input_files "$TOKENIZED_FILES" --output_dir $OUTPUT_DIR --num_words $NUM_WORDS --max_sentence_length 50 --case_sensitive False
