# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

from taacalgo.dao import model_dao
from taacalgo.model import train_model
from taacalgo.util import logger

# 导入数据 user
train_user = model_dao()
# log
logger.info('successful input train_user')

# 双层transformer
# log
logger.info('start train transformer')
train_model(train_user, 'tr',    6 , '+', 1,  epoch_=5)
train_model(train_user, 'tr',    16, '+', 2,  epoch_=5)
train_model(train_user, 'tr',    32, '+', 3,  epoch_=5)
train_model(train_user, 'tr',    64, '+', 4,  epoch_=5)
train_model(train_user, 'tr',    6,  '-', 5,  epoch_=5)
train_model(train_user, 'tr',    16, '-', 6,  epoch_=5)
train_model(train_user, 'tr',    32, '-', 7,  epoch_=5)
train_model(train_user, 'tr',    64, '-', 8,  epoch_=5)

# 单层transformer + 双层lstm
# log
logger.info('start train trlstm')
train_model(train_user, 'trlstm', 6,  '+', 9 , lr_=0.0005)
train_model(train_user, 'trlstm', 16, '+', 10, lr_=0.0005)
train_model(train_user, 'trlstm', 32, '+', 11, lr_=0.0005)
train_model(train_user, 'trlstm', 64, '+', 12, lr_=0.0005)
train_model(train_user, 'trlstm', 6,  '-', 13, lr_=0.0005)
train_model(train_user, 'trlstm', 16, '-', 14, lr_=0.0005)
train_model(train_user, 'trlstm', 32, '-', 15, lr_=0.0005)
train_model(train_user, 'trlstm', 64, '-', 16, lr_=0.0005)

# 单层transformer + 双层lstm advertiser可训练
# log
logger.info('start train trlstm advertiser trainable')
train_model(train_user, 'trlstm', 6,  '+', 17, freeze_adver=False)
train_model(train_user, 'trlstm', 16, '+', 18, freeze_adver=False)
train_model(train_user, 'trlstm', 32, '+', 19, freeze_adver=False)
train_model(train_user, 'trlstm', 64, '+', 20, freeze_adver=False)
train_model(train_user, 'trlstm', 6,  '-', 21, freeze_adver=False)
train_model(train_user, 'trlstm', 16, '-', 22, freeze_adver=False)
train_model(train_user, 'trlstm', 32, '-', 23, freeze_adver=False)
train_model(train_user, 'trlstm', 64, '-', 24, freeze_adver=False)
# log
logger.info('successful train model')