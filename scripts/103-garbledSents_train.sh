#!/usr/bin/env bash

export BERT_BASE_DIR=../models/cased_L-12_H-768_A-12
export DATA_DIR=../glue

python3 ../run_classifier.py \
  --task_name=garbledSents \
  --do_train=False \
  --do_eval=True \
  --do_predict=False \
  --data_dir=$DATA_DIR/garbledSents \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=64 \
  --predict_batch_size=64 \
  --learning_rate=1e-5 \
  --num_train_epochs=12 \
  --num_gpu_cores=2 \
  --do_lower_case=False \
  --output_dir=../fine_tuned/garbledSents \
  | tee garbledSents.log
