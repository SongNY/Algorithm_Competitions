# -*- coding: utf-8 -*-
# @author:
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import pandas as pd
from .util import path_traindata_prelimnary, path_traindata_semifinal, path_testdata

def w2v_dao():
    """ w2v 导入数据: 训练集1 & 训练集2 & 测试集 分别先与ad merge 后全部concat

    :return click: 全数据集点击序列

    """
    trainpath = path_traindata_prelimnary
    trainpath2 = path_traindata_semifinal
#     testpath = path_testdata

    train_click=pd.read_csv(trainpath / 'click_log.csv')
    train2_click=pd.read_csv(trainpath2 / 'click_log.csv')
#     test_click = pd.read_csv(testpath / 'click_log.csv')

    train_ad=pd.read_csv(trainpath / 'ad.csv')
    train2_ad=pd.read_csv(trainpath2 / 'ad.csv')
#     test_ad=pd.read_csv(testpath / 'ad.csv')

#     test_click=test_click.merge(test_ad,how='left')
    train_click=train_click.merge(train_ad,how='left')
    train2_click=train2_click.merge(train2_ad,how='left')

#     click=pd.concat([train_click,train2_click,test_click])
    click=pd.concat([train_click,train2_click])
    click=click.sort_values(['user_id','time'])

    return click


def model_dao():
    """ model 导入数据: 训练集1 & 训练集2 user

    :return train_user: 全数据集点击序列

    """
    trainpath = path_traindata_prelimnary
    trainpath2 = path_traindata_semifinal
    train_user = pd.read_csv(trainpath / 'user.csv')
    train2_user = pd.read_csv(trainpath2 / 'user.csv')
    train_user = pd.concat([train_user, train2_user], ignore_index=True)

    # 转换为20分类
    m = 0
    for j in range(1, 3):
        for i in range(1, 11):
            train_user.loc[(train_user.age == i) & (train_user.gender == j), 'label'] = int(m)
            m = m + 1
    train_user['label'] = train_user.label.astype(int)

    return train_user