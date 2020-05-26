import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df_history_action = pd.read_pickle('../user_data/tmp_data/action_history.plk')
df_feature = pd.read_pickle('../user_data/tmp_data/base_feature.plk')
df_courier = pd.read_pickle('../user_data/tmp_data/courier.plk')
df_order = pd.read_pickle('../user_data/tmp_data/order.plk')
df_distance = pd.read_pickle('../user_data/tmp_data/distance.plk')

df_temp = df_history_action.groupby(['group'])['expect_time'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
df_temp.columns = ['group', 'current_time']
df_feature = df_feature.merge(df_temp, how='left')

df_temp = df_history_action.groupby(['group'])['tracking_id'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
df_temp.columns = ['group', 'last_tracking_id']
df_feature = df_feature.merge(df_temp, how='left')

df_temp = df_history_action.groupby(['group'])['action_type'].apply(
    lambda x: x.values.tolist()[-1]).reset_index()
df_temp.columns = ['group', 'last_action_type']
df_feature = df_feature.merge(df_temp, how='left')

df_distance = df_distance.rename(columns={'tracking_id': 'last_tracking_id',
                                          'source_type': 'last_action_type', 
                                          'target_tracking_id': 'tracking_id',
                                          'target_type': 'action_type'})
df_feature = df_feature.merge(df_distance.drop(
    ['courier_id', 'wave_index', 'date'], axis=1), how='left')

df_feature = df_feature.merge(
    df_order[['tracking_id', 'weather_grade', 'aoi_id', 'shop_id', 'promise_deliver_time',
              'estimate_pick_time']], how='left')

##courier
df_feature = df_feature.merge(df_courier, how='left')

##base
df_sum=df_feature.groupby(['group']).size().reset_index()
df_sum.columns = ['group','unfinished_num_sum']
df_feature = df_feature.merge(df_sum, how='left')

df_order['group']=df_order['date'].astype(
    'str') + df_order['courier_id'].astype('str') + df_order['wave_index'].astype('str')
df_sum=df_order.groupby(['group']).size().reset_index()
df_sum.columns = ['group','all_order_num']
df_feature = df_feature.merge(df_sum, how='left')
df_feature['finish_freq']=df_feature['unfinished_num_sum']/df_feature['all_order_num']/2

##distance
df_feature['delta_lng']=df_feature['source_lng']-df_feature['target_lng']
df_feature['delta_lat']=df_feature['source_lat']-df_feature['target_lat']
df_feature['delta_abs_lng']=abs(df_feature['source_lng']-df_feature['target_lng'])
df_feature['delta_abs_lat']=abs(df_feature['source_lat']-df_feature['target_lat'])
df_feature['delta_dis']=(df_feature['delta_lng']**2+df_feature['delta_lat']**2)**0.5

df_feature['lng_max']=0
id_index=df_feature.groupby('group')['delta_lng'].idxmax()
df_feature.loc[id_index,'lng_max']=1

df_feature['lng_min']=0
id_index=df_feature.groupby('group')['delta_lng'].idxmin()
df_feature.loc[id_index,'lng_min']=1

df_feature['lat_max']=0
id_index=df_feature.groupby('group')['delta_lat'].idxmax()
df_feature.loc[id_index,'lat_max']=1

df_feature['lat_min']=0
id_index=df_feature.groupby('group')['delta_lat'].idxmin()
df_feature.loc[id_index,'lat_min']=1

df_feature['latest_grid']=0
id_index=df_feature.groupby('group')['grid_distance'].idxmin()
df_feature.loc[id_index,'latest_grid']=1

##time
df_feature['delta_pick_time']=df_feature['estimate_pick_time']-df_feature['current_time']
df_feature['delta_deliver_time']=df_feature['promise_deliver_time']-df_feature['current_time']
df_feature['current_hour']=pd.to_datetime(df_feature['current_time'].values, utc=True, unit='s').tz_convert(
            "Asia/Shanghai").hour
df_feature['current_hour_bin']='other'
df_feature.loc[df_feature['current_hour'].isin([11,12,13]),'current_hour_bin']='lunch'
df_feature.loc[df_feature['current_hour'].isin([17,18,19]),'current_hour_bin']='dinner'

df_feature['latest_deliver']=0
id_index=df_feature.groupby('group')['delta_deliver_time'].idxmin()
df_feature.loc[id_index,'latest_deliver']=1

df_feature.to_pickle('../user_data/tmp_data/all_feature.plk')

print('特征工程完成')
print('======================================')