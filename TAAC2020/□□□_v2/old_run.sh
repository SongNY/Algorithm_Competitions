
cd `dirname $0`

cd src/jinzhen
bash run.sh

cd ../ningyu
bash run.sh

cd ..
python output_prediction.py
