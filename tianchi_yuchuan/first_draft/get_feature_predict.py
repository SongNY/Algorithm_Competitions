import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
from inspect import isfunction

warnings.filterwarnings('ignore')

def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        if isfunction(ag):
            if (ag==pd.Series.kurt):
                agg_dict[f'{target}_kurt'] = ag
            elif (ag==q_25):
                agg_dict[f'{target}_25'] = ag
            elif (ag==q_75):
                agg_dict[f'{target}_75'] = ag
            else:
                agg_dict[f'{target}_mode'] = ag
        else:
            agg_dict[f'{target}_{ag}'] = ag
    # print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def q_25(x):
    return x.quantile(q=0.25)
def q_75(x):
    return x.quantile(q=0.75)
def modex(x):
    return np.mean(pd.Series.mode(x))

def extract_feature(df, train):

    t = group_feature(df, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','x',['count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','median','std','skew','sum',q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d',['mean','std','skew','sum',q_75,pd.Series.kurt])
    train = pd.merge(train, t, on='ship', how='left')
    
    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')

    t=df[(df['v']==0)].groupby(['ship','x','y']).size().groupby('ship').idxmax().apply(pd.Series).iloc[:,1:]
    t.rename(columns={1:'x_0', 2:'y_0'},inplace = True)
    train = pd.merge(train, t, on='ship', how='left')
    
    train['x_0_x_max']=train['x_max']-train['x_0']
    train['x_0_x_mean']=train['x_0']-train['x_mean']
    train['x_0_x_min']=train['x_0']-train['x_min']
    train['x_0_x_25']=train['x_0']-train['x_25']

    train['y_0_y_max']=train['y_max']-train['y_0']
    train['y_0_y_mean']=train['y_0']-train['y_mean']
    train['y_0_y_min']=train['y_0']-train['y_min']
     
    return train

def extract_dt(df):
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    return df

def map_feature(train):
    m,n=3,3
    df=train
    ymax,ymin,xmax,xmin=df['y'].max(),df['y'].min(),df['x'].max(),df['x'].min()
    yy=(df['y']-ymin)/(ymax-ymin)
    xx=(df['x']-xmin)/(xmax-xmin)
    count,count10,vmean,dmean=[],[],[],[]
    for i in range(m):
        count.append([]),count10.append([]),vmean.append([]),dmean.append([])
        for j in range(n):
            if ((i==m)|(j==n)):
                dfij=df[((i/m)<=yy) & (yy<=((i+1)/m)) & ((j/n)<=xx) & (xx<=((j+1)/n))]
            else:
                dfij=df[((i/m)<=yy) & (yy<((i+1)/m)) & ((j/n)<=xx) & (xx<((j+1)/n))]
            cc=dfij.shape[0]
            count[i].append(cc)
            count10[i].append(cc!=0)
            vmean[i].append(dfij['v'].mean())
            dmean[i].append(dfij['d'].mean())
    map_f=np.array([np.array(count),np.array(count10),np.array(vmean),np.array(dmean)]).reshape(m*n*4).T
    return pd.DataFrame(map_f)

path=os.getcwd()
choose_map_feature=['5', '18', '19', '20', '22', '23','26', '29']
testA = pd.read_hdf(path+'/data/testB.h5')
testA = extract_dt(testA)
testA_label = testA.drop_duplicates('ship')
testA_label = extract_feature(testA, testA_label)
testA_m_f=testA.groupby(['ship']).apply(map_feature).unstack().fillna(0)
testA_m_f.columns=[str(i) for i in range(36)]
testA_m_f=testA_m_f[choose_map_feature]
testA_label=pd.merge(testA_label, testA_m_f, on='ship', how='left')

features = [x for x in testA_label.columns if x not in ['ship','type','time','diff_time','date','d','hour','weekday']]
target = 'type'
# print(len(features))
# print(np.array(features))

models=[]
for i in range(5):
    model = lgb.Booster(model_file=path+'/model/lgb'+str(i)+'.txt')
    models.append(model)

predA = np.zeros((len(testA_label),3))
for model in models:
    testA_pred = model.predict(testA_label[features])
    predA += testA_pred/5

type_map = dict(zip(['拖网', '围网', '刺网'], np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
    
pred = np.argmax(predA, axis=1)
sub = testA_label[['ship']]
sub['pred'] = pred

# print(sub['pred'].value_counts(1))
sub['pred'] = sub['pred'].map(type_map_rev)
sub.to_csv(path+'/result.csv', index=None, header=None)