#!/usr/bin/env bash

export BERT_BASE_DIR=../models/cased_L-24_H-1024_A-16
export DATA_DIR=../glue

nohup python3 ../run_classifier.py \
  --task_name=SED \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR/SED \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=64 \
  --learning_rate=2e-5 \
  --num_train_epochs=5 \
  --do_lower_case=False \
  --output_dir=../fine_tuned/sed \
  | tee run_classifier.log

tail -f run_classifier.log
