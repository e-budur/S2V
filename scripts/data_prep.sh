
NUM_WORDS=2196018
OUTPUT_DIR="../data/UMBC-SMALL/TFRecords" 
VOCAB_FILE="../dictionaries/glove/glove.840B.300d_dictionary.txt"  
TOKENIZED_FILES="../data/UMBC-SMALL/txt_tokenized/*"

python src/data/preprocess_dataset.py \
  --input_files "$TOKENIZED_FILES" \
  --vocab_file $VOCAB_FILE \
  --output_dir $OUTPUT_DIR \
  --num_words $NUM_WORDS \
  --max_sentence_length 50 \
  --case_sensitive
