#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU_NUM=0  # A gpu number to use

GPU_FRACTION=0.95
DATA_DIR= # SET YOURS
TMP_DIR= # SET YOURS
SESS_DIR= # SET YOURS
T2T_USR_DIR="./usr_dir"
DECODING_PY="./autoregressive_decode.py"
PROBLEM='algorithmic_scan_sep_length'
MODEL='gru_seq2seq'
HPARAMS_SET='gru_attention_scan'
TRAIN_STEPS=32000
SAVE_STEPS=8000
TRIAL=0
TEST_SHARDS=("0")
MODEL_DIR=${SESS_DIR}${PROBLEM}-${MODEL}-${HPARAMS_SET}.${TRIAL}
DECODE_TO_FILE=${MODEL_DIR}/decode_00000

t2t-datagen --problem=${PROBLEM} --t2t_usr_dir=${T2T_USR_DIR} --data_dir=${DATA_DIR} --tmp_dir=${TMP_DIR}


CUDA_VISIBLE_DEVICES=${GPU_NUM}, t2t-trainer --generate_data --data_dir=${DATA_DIR} --tmp_dir=${TMP_DIR} --output_dir=${MODEL_DIR} --t2t_usr_dir=${T2T_USR_DIR} --problem=${PROBLEM} --model=${MODEL} --hparams_set=${HPARAMS_SET} --train_steps=${TRAIN_STEPS} --local_eval_frequency=${SAVE_STEPS}


for TEST_SHARD in ${TEST_SHARDS[@]};
do
        CUDA_VISIBLE_DEVICES=${GPU_NUM} python ${DECODING_PY} --data_dir=${DATA_DIR} --problem=${PROBLEM} --model=${MODEL} --hparams_set=${HPARAMS_SET} --t2t_usr_dir=${T2T_USR_DIR} --model_dir=${MODEL_DIR} --test_shard=${TEST_SHARD} --global_steps=${TRAIN_STEPS} --gpu_fraction=${GPU_FRACTION} --decode_to_file=${DECODE_TO_FILE}
done
