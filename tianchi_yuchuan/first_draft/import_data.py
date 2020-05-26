import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

path=os.getcwd()
testB_path = path+'/data/hy_round1_testB_20200221'
testB_files = os.listdir(testB_path)

ret = []
for file in testB_files:
    df = pd.read_csv(f'{testB_path}/{file}')
    ret.append(df)
df = pd.concat(ret)
df.columns = ['ship','x','y','v','d','time']
df.to_hdf(path+'/data/testB.h5', 'df', mode='w')

del df,ret