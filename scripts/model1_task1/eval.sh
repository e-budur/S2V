WORK_DIR="/truba/home/ebudur/tse-s2v" #needs to be changed according to your specific environment

TASK='SICK'
#TASK='TREC'
#TASK='MSRP'
#TASK='MR'
#TASK='CR'
#TASK='SUBJ'
#TASK='MPQA'

MDLS_PATH="$WORK_DIR/data/results"
MDL_CFGS="$WORK_DIR/model_configs"
#GLOVE_PATH="$WORK_DIR/data/word_embeddings/glove"

#CFG="BS400-W620-S1200-case-bidir"
CFG="UMBC-SMALL"
#CFG="MC-UMBC"

SKIPTHOUGHTS="ST_dir"
DATA="$WORK_DIR/data/sem_sim/eng/SICK/"

export CUDA_VISIBLE_DEVICES=''
#export PYTHONPATH="$SKIPTHOUGHTS:$PYTHONPATH"
python /truba/home/ebudur/tse-s2v/src/evaluate.py \
	--eval_task=$TASK \
	--data_dir=$DATA \
	--model_config="$MDL_CFGS/$CFG/eval.json" \
	--results_path="$MDLS_PATH"

