#!/bin/bash

set -e
cd `dirname $0`

python ../feature/pre_data.py
python ../feature/get_feature.py
python ../model/model.py
python ../code/predict.py
