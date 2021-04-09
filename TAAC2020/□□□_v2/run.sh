
cd `dirname $0`

if [ ! -f "output.tar.gz" ];then
  wget https://taac-1252411451.cos.ap-guangzhou.myqcloud.com/output.tar.gz
fi

tar xzvf output.tar.gz

cd src
python get_final_submission.py

