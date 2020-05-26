import pandas as pd
import numpy as np
import lightgbm as lgb
import math
import os
import time
import xgboost as xgb

## 读取数据
path='/home/sunnyu/xmgjyh/c_data/'
x_train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
y_train = pd.read_csv(path+'train_target.csv')



##model1:xgb1原始特征
out_feature=['id','isNew']

xgb100 = xgb.XGBClassifier(n_jobs=20,n_estimators=100)
xgb1 = xgb100.fit(x_train.drop(columns=out_feature),y_train['target'])
pred_xgb1=xgb1.predict_proba(test.drop(columns=out_feature))[:,1]



##测试集训练集特征构造

## 持续
x_train['certValidStop'][x_train['certValidStop']>10**10]=6314112000+100*365*24*60*60
x_train['certlast']=x_train['certValidStop']-x_train['certValidBegin']
x_train['certlast_day']=(x_train['certlast']/60/60/24).astype(int)
x_train['certValidBegin_year']=x_train['certValidBegin'].apply(lambda x: time.localtime(x-2207520000)[0])
x_train['certValidStop_year']=x_train['certValidStop'].apply(lambda x: time.localtime(x-2207520000)[0])

## 地域特征
x_train['dist2']=x_train['dist'].astype(str).str.slice(0,2).astype(int)
x_train['dist4']=x_train['dist'].astype(str).str.slice(0,4).astype(int)

x_train['residentAddr'][x_train['residentAddr']==-999]=0
x_train['residentAddr2']=x_train['residentAddr'].astype(str).str.slice(0,2).astype(int)
x_train['residentAddr4']=x_train['residentAddr'].astype(str).str.slice(0,4).astype(int)

x_train['certId2']=x_train['certId'].astype(str).str.slice(0,2).astype(int)
x_train['certId4']=x_train['certId'].astype(str).str.slice(0,4).astype(int)

x_train['cd2']=(x_train['certId2']==x_train['dist2']).astype(int)
x_train['cd4']=(x_train['certId4']==x_train['dist4']).astype(int)
x_train['cd6']=(x_train['certId']==x_train['dist']).astype(int)

x_train['rcd2']=((x_train['certId2']==x_train['dist2']) & (x_train['certId2']==x_train['residentAddr2'])).astype(int)
x_train['rcd4']=((x_train['certId4']==x_train['dist4']) & (x_train['certId4']==x_train['residentAddr4'])).astype(int)
x_train['rcd6']=((x_train['certId']==x_train['dist']) & (x_train['certId']==x_train['residentAddr'])).astype(int)

## 银行卡
x_train['bankCard'].fillna(0,inplace=True)
x_train['bankCard'][x_train['bankCard']==-999]=0
x_train['bankCard6']=x_train['bankCard'].astype(int).astype(str).str.slice(0,6).astype(int)


## x统计

xname=list()
for i in range(79):
    xname.append('x_'+str(i))
x_fea=x_train[xname].apply([sum,np.std],axis=1)
x_train=pd.concat([x_train,x_fea],axis=1)
x_train['x0_num']=(x_train[xname]==0).apply(sum,axis=1)
x_train['sum'][x_train['sum']<0]=-1

del x_fea,xname

## 持续时间
test['certValidStop'][test['certValidStop']>10**10]=6314112000+100*365*24*60*60
test['certlast']=test['certValidStop']-test['certValidBegin']
test['certlast_day']=(test['certlast']/60/60/24).astype(int)
test['certValidBegin_year']=test['certValidBegin'].apply(lambda x: time.localtime(x-2207520000)[0])
test['certValidStop_year']=test['certValidStop'].apply(lambda x: time.localtime(x-2207520000)[0])


## 地域特征
test['dist2']=test['dist'].astype(str).str.slice(0,2).astype(int)
test['dist4']=test['dist'].astype(str).str.slice(0,4).astype(int)

test['residentAddr'][test['residentAddr']==-999]=0
test['residentAddr2']=test['residentAddr'].astype(str).str.slice(0,2).astype(int)
test['residentAddr4']=test['residentAddr'].astype(str).str.slice(0,4).astype(int)

test['certId2']=test['certId'].astype(str).str.slice(0,2).astype(int)
test['certId4']=test['certId'].astype(str).str.slice(0,4).astype(int)

test['cd2']=(test['certId2']==test['dist2']).astype(int)
test['cd4']=(test['certId4']==test['dist4']).astype(int)
test['cd6']=(test['certId']==test['dist']).astype(int)


test['rcd2']=((test['certId2']==test['dist2']) & (test['certId2']==test['residentAddr2'])).astype(int)
test['rcd4']=((test['certId4']==test['dist4']) & (test['certId4']==test['residentAddr4'])).astype(int)
test['rcd6']=((test['certId']==test['dist']) & (test['certId']==test['residentAddr'])).astype(int)



## 银行卡
test['bankCard'].fillna(99999999,inplace=True)
test['bankCard'][test['bankCard']==-999]=0
test['bankCard6']=test['bankCard'].astype(str).str.slice(0,6).astype(int)



## x统计

xname=list()
for i in range(79):
    xname.append('x_'+str(i))
x_fea=test[xname].apply([sum,np.std],axis=1)
test=pd.concat([test,x_fea],axis=1)
test['x0_num']=(test[xname]==0).apply(sum,axis=1)
test['sum'][test['sum']<0]=-1

del x_fea,xname


##model2:xgb2新特征
weight=(y_train['target']==0).sum()/(y_train['target']==1).sum()
out_feature=['id','isNew','certId','residentAddr','dist','bankCard','certValidStop','certValidBegin','certlast']
xgb70 = xgb.XGBClassifier(n_jobs=20,
                           scale_pos_weight=weight,
                           n_estimators=70,
                           max_depth=3
                          )
xgb2 = xgb70.fit(x_train.drop(columns=out_feature),y_train['target'])
pred_xgb2 = xgb2.predict_proba(test.drop(columns=out_feature))[:,1]



##model3:lgb新特征
cat_feature=['edu','highestEdu','loanProduct','linkRela','gender','weekday','unpayNormalLoan','job','setupHour','ethnic',
                   'ncloseCreditCard','unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan','5yearBadloan',
                   'certId2','residentAddr2','dist2','bankCard6','cd2','cd4','cd6','rcd2','rcd4','rcd6']
out_feature=['id','isNew','certId','residentAddr','dist','bankCard','certId4','residentAddr4','dist4','certValidStop','certValidBegin','certlast']
for i in range(79):
    cat_feature.append('x_'+str(i))
    
    
train_data=lgb.Dataset(x_train.drop(columns=out_feature), 
#                       categorical_feature=cat_feature,
                       label=y_train['target'])

param = { 
        'objective':'binary',
        'is_unbalance':'true',
        'metric':'auc',
        'boosting':'dart',
        'num_leaves':3,
       'n_job':20,
#          'learning_rate': 0.08,
         'min_data_in_leaf':2**10,
         'neg_bagging_fraction':0.8,
         'feature_fraction':0.7,
          'lambda_l1':0.5,
          'lambda_l2':0.5,
#         'bagging_freq': 5,
#         'max_bin':2**6,
#         'min_data_in_bin':2**7,
#         'verbose': 0     
}


bst=lgb.train(param, 
              train_data,
              num_boost_round=900)
pred_lgb=bst.predict(test.drop(columns=out_feature))


## 融合
pred=0.618*pred_xgb1/max(pred_xgb1)+pred_lgb*0.392
pred=0.392*pred/max(pred)+pred_xgb2*0.618/max(pred_xgb2)
pred=pred/max(pred)
sub=pd.DataFrame([test['id'],pred],index=['id','target']).T
sub.to_csv('./submission_final.csv',index=None)