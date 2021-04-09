
cd `dirname $0`
python preprocess.py
python preword2vec.py

cd word2vec
make
bash train.sh
bash train512.sh
bash train512_min4.sh

cd ..
python model_main.py lstm1
python model_main.py trlstm1
python model_main.py trlstm2
python model_main.py dnn1
python model_main.py dnn2
