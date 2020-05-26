import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
from inspect import isfunction
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


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
    print(agg_dict)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

def q_25(x):
    return x.quantile(q=0.25)
def q_75(x):
    return x.quantile(q=0.75)
def modex(x):
    return np.mean(pd.Series.mode(x))
def corrxy(x):
    return train['x'].corr(train['y'])

def extract_feature(df, train):
    
    t = group_feature(df, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','x',['count'])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','v',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','d',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')

    df['diffx']=df['x'].diff()
    df['diffy']=df['y'].diff()
    df['diffv']=df['v'].diff()
    df['diffd']=df['d'].diff()
    df['difflen']=(df['diffx']**2+df['diffy']**2)**0.5
    t = group_feature(df, 'ship','diffx',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','diffy',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','diffv',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','diffd',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','difflen',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    
    t=df[(df['v']==0)].groupby(['ship','x','y']).size().groupby('ship').idxmax().apply(pd.Series).iloc[:,1:]
    t.rename(columns={1:'x_0', 2:'y_0'},inplace = True)
    train = pd.merge(train, t, on='ship', how='left')
    
    train['x_0_x_max']=train['x_max']-train['x_0']
    train['x_0_x_mean']=train['x_0']-train['x_mean']
    train['x_0_x_min']=train['x_0']-train['x_min']
    train['x_0_x_25']=train['x_0']-train['x_25']
    train['x_0_x_median']=train['x_0']-train['x_median']
    train['x_0_x_75']=train['x_0']-train['x_75']
    train['x_0_x_mode']=train['x_0']-train['x_mode']

    train['y_0_y_max']=train['y_max']-train['y_0']
    train['y_0_y_mean']=train['y_0']-train['y_mean']
    train['y_0_y_min']=train['y_0']-train['y_min']
    train['y_0_y_25']=train['y_0']-train['y_25']
    train['y_0_y_median']=train['y_0']-train['y_median']
    train['y_0_y_75']=train['y_0']-train['y_75']
    train['y_0_y_mode']=train['y_0']-train['y_mode']

    train['x_max_x_min'] = train['x_max'] - train['x_min']
    train['y_max_y_min'] = train['y_max'] - train['y_min']
    
    train['y_max_x_min'] = train['y_max'] - train['x_min']
    train['y_max_x_max'] = train['y_max'] - train['x_max']
    train['x_max_y_min'] = train['x_max'] - train['y_min']
    train['x_min_y_min'] = train['x_min'] - train['y_min']
    
    train['slope'] = train['y_max_y_min'] / np.where(train['x_max_x_min']==0, 0.001, train['x_max_x_min'])
    train['area'] = train['x_max_x_min'] * train['y_max_y_min']

    t = df.groupby('ship')['time'].agg({'diff_time':lambda x:np.max(x)-np.min(x)}).reset_index()
    t['diff_second'] = t['diff_time'].dt.seconds
    train = pd.merge(train, t, on='ship', how='left')
    
#     df['v_cos'] = df['v'] * np.cos(df['d'])
#     df['v_sin'] = df['v'] * np.sin(df['d'])
#     t = group_feature(df, 'ship','v_cos',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df, 'ship','v_sin',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     train = pd.merge(train, t, on='ship', how='left')  
    
    df['k']=df['y']/df['x']
    df['b']=df['y']-df['k'].mean()*df['x']
    t = group_feature(df, 'ship','k',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df, 'ship','b',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    train = pd.merge(train, t, on='ship', how='left')
    
    df_night=df[(df['hour']<6) | (df['hour']>=18)]
    t = group_feature(df_night, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_night' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_night, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_night' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_night, 'ship','v',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_night' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_night, 'ship','d',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_night' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    
    df_day=df[(df['hour']>=6) & (df['hour']<18)]    
    t = group_feature(df_day, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_day' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_day, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_day' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_day, 'ship','v',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_day' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    t = group_feature(df_day, 'ship','d',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
    t.columns=['ship']+[x+'_day' for x in t.columns[1:].tolist()]
    train = pd.merge(train, t, on='ship', how='left')
    
#     v_medain_list=np.repeat(df.groupby(['ship'])['v'].median().values,df.groupby(['ship'])['v'].count().values, axis=0)
#     df_vsmall=df[df['v']<v_medain_list]
#     t = group_feature(df_vsmall, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vsamll' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vsmall, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vsamll' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vsmall, 'ship','v',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vsamll' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vsmall, 'ship','d',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vsamll' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')

#     df_vlarge=df[df['v']>v_medain_list]
#     t = group_feature(df_vlarge, 'ship','x',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vlarge' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vlarge, 'ship','y',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vlarge' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vlarge, 'ship','v',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vlarge' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
#     t = group_feature(df_vlarge, 'ship','d',['max','min','median','mean','std','skew','sum',q_25,q_75,pd.Series.kurt,modex])
#     t.columns=['ship']+[x+'_vlarge' for x in t.columns[1:].tolist()]
#     train = pd.merge(train, t, on='ship', how='left')
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
    count,count10,vmean,dmean,vstd,dstd,xstd,ystd=[],[],[],[],[],[],[],[]
    for i in range(m):
        count.append([]),count10.append([]),vmean.append([]),dmean.append([]),dstd.append([]),vstd.append([]),xstd.append([]),ystd.append([])
        for j in range(n):
            if ((i==m)|(j==n)):
                dfij=df[((i/m)<=yy) & (yy<=((i+1)/m)) & ((j/n)<=xx) & (xx<=((j+1)/n))]
            else:
                dfij=df[((i/m)<=yy) & (yy<((i+1)/m)) & ((j/n)<=xx) & (xx<((j+1)/n))]
            cc=dfij.shape[0]
            count[i].append(cc)
            count10[i].append(cc!=0)
            xstd[i].append(dfij['x'].std())
            ystd[i].append(dfij['y'].std())
            vmean[i].append(dfij['v'].mean())
            vstd[i].append(dfij['v'].std())
            dmean[i].append(dfij['d'].mean())
            dstd[i].append(dfij['d'].std())
    map_f=np.array([np.array(count),np.array(count10),np.array(vmean),np.array(dmean),
                    np.array(vstd),np.array(dstd),np.array(xstd),np.array(ystd)]).reshape(m*n*8).T
    return pd.DataFrame(map_f)

path=os.getcwd()
train = pd.read_hdf(path+'/data/train.h5')
train = extract_dt(train)
train_label = train.drop_duplicates('ship')
type_map = dict(zip(['拖网', '刺网', '围网'], np.arange(3)))
type_map_rev = {v:k for k,v in type_map.items()}
train_label['type'] = train_label['type'].map(type_map)
train_label = extract_feature(train, train_label)
train_m_f=train.groupby(['ship']).apply(map_feature).unstack()
train_m_f.columns=[str(i) for i in range(72)]
train_label=pd.merge(train_label, train_m_f, on='ship', how='left')
train_co=train.groupby('ship').apply(lambda x :x['x'].corr(x['y'])).reset_index()
train_co.columns=['ship','corr']
train_label=pd.merge(train_label, train_co, on='ship', how='left')

features=pd.read_csv(path+'/model/features.csv',header=None)
features=features[0].values.tolist()
target = 'type'
print(len(features))
print(np.array(features))


params = {
    'num_leaves':40,
    'n_estimators':2500,
    'boosting_type': 'dart',
    'objective': 'multiclassova',
    'num_class': 3,
    'num_threads':20,
}

fold = StratifiedKFold(n_splits=6, shuffle=True, random_state=513)

X = train_label[features].copy()
y = train_label[target]
models = []
a40=[900,1450,1370,630,750,610]
i=0

def mac_f1(predss,mm):
    labell=mm.get_label().values
    predss=np.argmax(predss.reshape(3,len(labell)), axis=0)
    f11=metrics.f1_score(labell,predss,average='macro')
    return 'f1-score',f11,True

oof = np.zeros((len(X), 3))
for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
    X_train = pd.concat([X.iloc[train_idx],X.iloc[train_idx][train_label.iloc[train_idx]['type']==1]])
    y_train = pd.concat([y.iloc[train_idx],y.iloc[train_idx][train_label.iloc[train_idx]['type']==1]])   
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    params['n_estimators']=a40[i]
    i+=1
    model = lgb.train(params, train_set, valid_sets=val_set,feval=mac_f1, verbose_eval=100)
    models.append(model)

for i in range(6):
    models[i].save_model(path+'/model/lgb'+str(i)+'.txt')
np.save(path+'/model/type_map.npy', type_map) 
print('finished train')
print(type_map)