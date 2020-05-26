#!/bin/bash

set -m

echo ">>>> Starting Bert Service... (Please wating for 60s)"
python start-bert-as-service.py -model_dir ../user_data/model_data/chinese_L-12_H-768_A-12 -num_worker=2 -max_seq_len=NONE 1>/dev/null 2>/dev/null &
sleep 60
python train.py
python main.py