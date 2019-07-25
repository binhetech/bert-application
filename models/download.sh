#!/usr/bin/env bash

if [ ! -d cased_L-12_H-768_A-12 ]; then
    # download pre-training model: BERT-Base Cased
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
    # unzip
    unzip cased_L-12_H-768_A-12.zip
fi

if [ ! -d cased_L-24_H-1024_A-16 ]; then
    # download pre-training model: BERT-Large Cased
    wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip
    # unzip
    unzip cased_L-24_H-1024_A-16.zip
fi