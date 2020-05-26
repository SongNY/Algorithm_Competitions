import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.utils import shuffle
import copy

df_feature=pd.read_pickle('../user_data/tmp_data/all_feature.plk')

###choose feature
df_feature_base_name=['wave_index','action_type','unfinished_num_sum','last_action_type','weather_grade']
df_feature_distance_name=['delta_lng','delta_lat','delta_abs_lng','delta_abs_lat','delta_dis','grid_distance',
                         'latest_grid','lat_max','lat_min','lng_min','lng_max']
df_feature_time_name=['delta_pick_time','delta_deliver_time','latest_deliver','current_hour','current_hour_bin']
df_feature_courier_name=['level','speed','max_load']
df_feature_name=df_feature_base_name+df_feature_distance_name+df_feature_time_name+df_feature_courier_name

for f in df_feature.select_dtypes('object'):
    if f not in ['date', 'type','group']:
        lbl = LabelEncoder()
        df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
        
##划分训练集 打乱顺序
df_testA = df_feature[df_feature['type'] == 'testA'].copy()
df_test = df_feature[df_feature['type'] == 'testB'].copy()
df_train = df_feature[df_feature['type'] == 'train'].copy()
df_train = shuffle(df_train, random_state=513)

params = {
    'num_leaves':31,
    'n_estimators': 200,
    'boosting_type': 'gbdt',
    'is_unbalance':'true',
    'objective': 'binary'
}
train_set = lgb.Dataset(df_train[df_feature_name], df_train['target'],free_raw_data=False)
model_final = lgb.train(params,train_set,
                        categorical_feature=['weather_grade','current_hour_bin'])

##预测testA 加入训练集
df_testA['pred']=model_final.predict(df_testA[df_feature_name])
same_track_id=df_testA.groupby('tracking_id').size()[df_testA.groupby('tracking_id').size()==2].reset_index()['tracking_id'].values.tolist()
df_testA_afterchoice=df_testA[~((df_testA['tracking_id'].isin(same_track_id)) & (df_testA['action_type']==0))]
maxid=df_testA_afterchoice.groupby('group')['pred'].idxmax()
df_testA['target']=0
df_testA.loc[maxid,'target']=1
df_train_testA=pd.concat([df_train,df_testA])

##train+testA model
params = {
    'num_leaves':31,
    'n_estimators': 200,
    'boosting_type': 'gbdt',
    'is_unbalance':'true',
    'objective': 'binary'
}
train_set = lgb.Dataset(df_train_testA[df_feature_name], df_train_testA['target'])
model_final = lgb.train(params,train_set,
                        categorical_feature=['weather_grade','current_hour_bin'],)
model_final.save_model('../user_data/model_data/model_pred.txt')


time_train=copy.deepcopy(df_train[df_train['target']==1])
time_train['detla_time']=time_train['expect_time']-time_train['current_time']
param = {
    'num_leaves':31,
    'n_estimators': 10000,
    'boosting_type': 'gbdt',
    'objective': 'mae',
    'metrics':'mae',
    'early_stopping_rounds': 50
}
train_time_set = lgb.Dataset(time_train[df_feature_name],time_train['detla_time'])
model_time_cv=lgb.cv(param,train_time_set,nfold=5,metrics='mae',
                     categorical_feature=['current_hour_bin','weather_grade'])
n_east=len(model_time_cv['l1-mean'])

param = {
    'num_leaves':31,
    'n_estimators': n_east,
    'boosting_type': 'gbdt',
    'objective': 'mae'
}
model_time=lgb.train(param,train_time_set,
                     categorical_feature=['current_hour_bin','weather_grade'])
model_time.save_model('../user_data/model_data/model_time.txt')
print('模型训练完成')
print('======================================')