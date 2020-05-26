import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
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
        
df_test = df_feature[df_feature['type'] == 'testB'].copy()
model_final = lgb.Booster(model_file='../user_data/model_data/model_pred.txt')
model_time = lgb.Booster(model_file='../user_data/model_data/model_time.txt')

pred_time=model_time.predict(df_test[df_feature_name])
df_test['expect_time']=df_test['current_time']+pred_time
pred=model_final.predict(df_test[df_feature_name])
df_test['pred']=pred

same_track_id=df_test.groupby('tracking_id').size()[df_test.groupby('tracking_id').size()==2].reset_index()['tracking_id'].values.tolist()
df_test_afterchoice=df_test[~((df_test['tracking_id'].isin(same_track_id)) & (df_test['action_type']==0))]
maxid=df_test_afterchoice.groupby('group')['pred'].idxmax()
df_sub=copy.deepcopy(df_test_afterchoice.loc[maxid])
df_sub.loc[df_sub['action_type']==0,'action_type']='DELIVERY'
df_sub.loc[df_sub['action_type']==1,'action_type']='PICKUP'
prediction=df_sub[['courier_id','wave_index','tracking_id','courier_wave_start_lng','courier_wave_start_lat','action_type','expect_time','date']]

for date in prediction['date'].unique():
    df_temp = prediction[prediction['date'] == date]
    del df_temp['date']
    df_temp.to_csv('../action_predict/action_{}.txt'.format(date), index=False)
    
print('预测完成')