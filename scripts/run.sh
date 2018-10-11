RESULTS_HOME="results"
MDL_CFGS="model_configs"
GLOVE_PATH=""

DATA_DIR="data/UMBC-SMALL/TFRecords"
NUM_INST=6063131 # Number of sentences

CFG="MC-UMBC-SMALL"

BS=400
SEQ_LEN=30

export CUDA_VISIBLE_DEVICES=0
python src/train.py \
	--results_path="$RESULTS_HOME/$CFG" \
    --input_file_pattern="$DATA_DIR/train-00000-of-00100" \
    --train_dir="$RESULTS_HOME/$CFG/train" \
    --learning_rate_decay_factor=0 \
    --batch_size=$BS \
    --sequence_length=$SEQ_LEN \
    --nepochs=1 \
    --num_train_inst=$NUM_INST \
    --save_model_secs=1800 \
    --Glove_path=$GLOVE_PATH \
    --model_config="$MDL_CFGS/$CFG/train.json" &

