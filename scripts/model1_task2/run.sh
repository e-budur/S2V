WORK_DIR="/truba/home/ebudur/tse-s2v" #needs to be changed according to your specific environment
RESULTS_HOME="$WORK_DIR/data/results"
MDL_CFGS="$WORK_DIR/model_configs"
#GLOVE_PATH="$WORK_DIR/data/word_embeddings/glove"

DATA_DIR="$WORK_DIR/data/bulk_sentences/en/UMBC-SMALL-TFRecords"
NUM_INST=1150000 # Number of sentences

CFG="UMBC-SMALL"

BS=400
SEQ_LEN=30

export CUDA_VISIBLE_DEVICES=0
python /truba/home/ebudur/tse-s2v/src/train.py \
	--results_path="$RESULTS_HOME/$CFG" \
    --input_file_pattern="$DATA_DIR/train-?????-of-00100" \
    --train_dir="$RESULTS_HOME/$CFG/train" \
    --learning_rate_decay_factor=0 \
    --batch_size=$BS \
    --sequence_length=$SEQ_LEN \
    --nepochs=1 \
    --num_train_inst=$NUM_INST \
    --save_model_secs=1800 \
    --Glove_path=$GLOVE_PATH \
    --model_config="$MDL_CFGS/$CFG/train.json" &

