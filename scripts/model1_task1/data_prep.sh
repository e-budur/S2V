
NUM_WORDS=100001
OUTPUT_DIR="../../data/bulk_sentences/en/UMBC-SMALL-TFRecords" 
#VOCAB_FILE="../../data/word_embeddings/glove/glove.840B.300d_dictionary.txt"  
TOKENIZED_FILES="../../data/bulk_sentences/en/UMBC-SMALL/*.txt"

python ../../src/data/preprocess_dataset.py \
  --input_files "$TOKENIZED_FILES" \
  --output_dir $OUTPUT_DIR \
  --num_words $NUM_WORDS \
  --max_sentence_length 50 \
  --case_sensitive False
