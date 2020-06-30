#!/usr/bin/env bash

export BERT_BASE_DIR=../models/cased_L-12_H-768_A-12
export DATA_DIR=../glue

nohup python3 ../run_classifier_imbalance.py \
  --task_name=DISC \
  --do_train=true \
  --do_eval=true \
  --do_predict=True \
  --max_steps_without_increase 10000 \
  --min_steps 1000 \
  --do_early_stopping=True \
  --class_weight=None \
  --data_dir=$DATA_DIR/DISC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=64 \
  --predict_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=5 \
  --num_gpu_cores=2 \
  --do_lower_case=False \
  --output_dir=../fine_tuned/disc |
  tee disc.log

tail -f disc.log
