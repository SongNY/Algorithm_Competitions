import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm

train_path = '../data/eleme_round1_train'
test_path = '../data/eleme_round1_testA'
testB_path = '../data/eleme_round1_testB'

courier_list = []
for f in os.listdir(os.path.join(train_path, 'courier')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(train_path, 'courier', f))
    df['date'] = date
    df['type'] = 'train'
    courier_list.append(df)
for f in os.listdir(os.path.join(test_path, 'courier')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(test_path, 'courier', f))
    df['date'] = date
    df['type'] = 'testA'
    courier_list.append(df)
for f in os.listdir(os.path.join(testB_path, 'courier')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(testB_path, 'courier', f))
    df['date'] = date
    df['type'] = 'testB'
    courier_list.append(df)
df_courier = pd.concat(courier_list, sort=False)
df_courier.to_pickle('../user_data/tmp_data/courier.plk')

order_list = []
for f in os.listdir(os.path.join(train_path, 'order')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(train_path, 'order', f))
    df['date'] = date
    df['type'] = 'train'
    order_list.append(df)
for f in os.listdir(os.path.join(test_path, 'order')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(test_path, 'order', f))
    df['date'] = date
    df['type'] = 'testA'
    order_list.append(df)
for f in os.listdir(os.path.join(testB_path, 'order')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(testB_path, 'order', f))
    df['date'] = date
    df['type'] = 'testB'
    order_list.append(df)
df_order = pd.concat(order_list, sort=False)
df_order.to_pickle('../user_data/tmp_data/order.plk')

distance_list = []
for f in os.listdir(os.path.join(train_path, 'distance')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(train_path, 'distance', f))
    df['date'] = date
    df['type'] = 'train'
    distance_list.append(df)
for f in os.listdir(os.path.join(test_path, 'distance')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(test_path, 'distance', f))
    df['date'] = date
    df['type'] = 'testA'
    distance_list.append(df)
for f in os.listdir(os.path.join(testB_path, 'distance')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(testB_path, 'distance', f))
    df['date'] = date
    df['type'] = 'testB'
    distance_list.append(df)
df_distance = pd.concat(distance_list, sort=False)
df_distance['group'] = df_distance['date'].astype(
    'str') + df_distance['courier_id'].astype('str') + df_distance['wave_index'].astype('str')
df_distance.to_pickle('../user_data/tmp_data/distance.plk')

df_actions = []
for f in os.listdir(os.path.join(train_path, 'action')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(train_path, 'action', f))
    df['date'] = date
    df['type'] = 'train'
    df_actions.append(df)
for f in os.listdir(os.path.join(test_path, 'action')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(test_path, 'action', f))
    df['date'] = date
    df['type'] = 'testA'
    df_actions.append(df)
for f in os.listdir(os.path.join(testB_path, 'action')):
    date = f.split('.')[0].split('_')[1]
    df = pd.read_csv(os.path.join(testB_path, 'action', f))
    df['date'] = date
    df['type'] = 'testB'
    df_actions.append(df)

ratio = 0.5
def read_feat(df):
    label_list = []
    history_list = []
    type = df['type'].values[0]
    groups = df.groupby(['courier_id', 'wave_index'])
    for name, group in tqdm(groups):
        if type == 'train':
            if group.shape[0]==4:
                label_data = group.tail(int(group.shape[0]-1))
                history_data = group.drop(label_data.index)
            else:
                label_data = group.tail(int(group.shape[0] * ratio))
                history_data = group.drop(label_data.index)
            label_data['target'] = 0
            label_data.reset_index(drop=True, inplace=True)
            label_data.loc[0, 'target'] = 1
            label_list.append(label_data)
            history_list.append(history_data)
        else:
            label_data = group[group['expect_time'] == 0]
            history_data = group.drop(label_data.index)
            label_data['target'] = None
            label_list.append(label_data)
            history_list.append(history_data)
    return pd.concat(label_list, sort=False), pd.concat(history_list, sort=False)
    
res = Parallel(n_jobs=12)(delayed(read_feat)(df) for df in tqdm(df_actions))
df_feature = [item[0] for item in res]
df_history = [item[1] for item in res]

df_feature = pd.concat(df_feature, sort=False)
df_history = pd.concat(df_history, sort=False)

df_feature['group'] = df_feature['date'].astype(
    'str') + df_feature['courier_id'].astype('str') + df_feature['wave_index'].astype('str')
df_history['group'] = df_history['date'].astype(
    'str') + df_history['courier_id'].astype('str') + df_history['wave_index'].astype('str')
df_feature['target'] = df_feature['target'].astype('float')
df_feature['id'] = range(df_feature.shape[0])

df_history.to_pickle('../user_data/tmp_data/action_history.plk')
df_feature.to_pickle('../user_data/tmp_data/base_feature.plk')

print('数据预处理完成')
print('======================================')