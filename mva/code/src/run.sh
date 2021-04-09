#!/bin/bash

set -e
cd `dirname $0`

python run_w2v.py
python run_model.py