#!/bin/bash

set -e
cd `dirname $0`

python import_data.py
python get_feature_predict.py