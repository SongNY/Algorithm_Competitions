import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

path=os.getcwd()
testB_path = path+'/tcdata/hy_round2_testB_20200312'
testB_files = os.listdir(testB_path)

ret = []
for file in testB_files:
    df = pd.read_csv(f'{testB_path}/{file}',engine='python' )
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time']
df.to_hdf(path+'/data/testB.h5', 'df', mode='w')

train_path = path+'/tcdata/hy_round2_train_20200225'
train_files = os.listdir(train_path)

ret = []
for file in train_files:
    df = pd.read_csv(f'{train_path}/{file}',engine='python' )
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time','type']
df.to_hdf(path+'/data/train.h5', 'df', mode='w')

del df,ret