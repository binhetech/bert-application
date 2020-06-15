#!/usr/bin/env bash

export BERT_BASE_DIR=../models/uncased_L-12_H-768_A-12
export DATA_DIR=../glue

nohup python3 ../run_classifier.py \
  --task_name=SED \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/SED \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --num_gpu_cores=1 \
  --do_lower_case=False \
  --output_dir=../fine_tuned/sed \
  | tee run_classifier.log

tail -f run_classifier.log
