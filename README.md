# Annotation of targets using parsing trees
Official implementation of "[Compositional Generalization via Parsing Tree Annotation]()" (IEEE ACCESS 2021) by Segwang Kim, Joonyoung Kim, and Kyomin Jung.

To make stanadard seq2seq models, such as Transformers or RNN seq2seq models, achieve compositional generalization on the SCAN length and MCD splits, we invent data augmentation technique using psrsing trees.

Example: Run scan.sh

Our implementation is based on [tensor2tensor](https://github.com/tensorflow/tensor2tensor) library.


## Requirements
```
tensorflow==1.13 (tested on cuda-10.0, cudnn-7.6)
python==3.6
tensor2tensor==1.13.1
tensorflow-probability==0.6.0
mesh-tensorflow==0.0.5
nltk
pandas
```

The followings commands are for generating data, training a model, and evaluating the model.

Refer to ``example.sh`` for variable settings, e.g., PROBLEM, HPARAMS_SET, MODEL and so on.

## Generate Data and Train a Model
```
CUDA_VISIBLE_DEVICES=${GPU_NUM}, t2t-trainer 
            --generate_data \
            --data_dir=${DATA_DIR} \
            --tmp_dir=${TMP_DIR} \
            --output_dir=${MODEL_DIR} \
            --t2t_usr_dir=${T2T_USR_DIR} \
            --problem=${PROBLEM} \
            --model=${MODEL} \
            --hparams_set=${HPARAMS_SET} \
            --train_steps=${TRAIN_STEPS} 
 ```
  
## Decode
```
DECODING_PY="autoregressive_decode.py"
CUDA_VISIBLE_DEVICES=${GPU_NUM} python ${DECODING_PY} 
                      --data_dir=${DATA_DIR} \
                      --problem=${PROBLEM} \
                      --model=${MODEL} \
                      --hparams_set=${HPARAMS_SET} \
                      --t2t_usr_dir=${T2T_USR_DIR} \
                      --model_dir=${MODEL_DIR} \
                      --test_shard=${TEST_SHARD} \
                      --global_steps=${TRAIN_STEPS} 
                      --decode_to_file=${DECODE_TO_FILE} 
 ```
