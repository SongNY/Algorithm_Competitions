# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/12/30

from taacalgo.dao import model_dao
from taacalgo.model import train_model
from taacalgo.util import logger

# 导入数据 user
train_user = model_dao()
# log
logger.info('successful input train_user')

for seed in range(2020,2025):
    
    logger.info('start train no_seq')
    try:
        train_model(train_user, 'no_seq', seed, dnn_dim=0)
    except Exception as e:
        error_msg = f"Error in train tr:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train dnn')
    try:
        train_model(train_user, 'dnn', seed)
    except Exception as e:
        error_msg = f"Error in train tr:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train rnn') 
    try:
        train_model(train_user, 'rnn', seed, rnn_layer=1, lr_=0.0002)
    except Exception as e:
        error_msg = f"Error in train rnn:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train gru') 
    try:
        train_model(train_user, 'gru', seed, gru_layer=1, lr_=0.0002)
    except Exception as e:
        error_msg = f"Error in train gru:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train lstm1')
    try:
        train_model(train_user, 'lstm', seed, lstm_layer=1, lr_=0.0005)
    except Exception as e:
        error_msg = f"Error in train lstm:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train lstm2')
    try:
        train_model(train_user, 'lstm', seed, lstm_layer=2, lr_=0.0005)
    except Exception as e:
        error_msg = f"Error in train lstm:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train transformer')
    try:
        train_model(train_user, 'tr', seed, tr_layer=1, tr_head=8, lr_=0.0001)
    except Exception as e:
        error_msg = f"Error in train tr:{str(e)}"
        logger.error(error_msg)
        
    logger.info('start train transformer2')
    try:
        train_model(train_user, 'tr', seed, tr_layer=2, tr_head=8, lr_=0.0001)
    except Exception as e:
        error_msg = f"Error in train tr:{str(e)}"
        logger.error(error_msg)

    logger.info('start train trlstm') 
    try:
        train_model(train_user, 'trlstm', seed, tr_layer=1, lstm_layer=1, lr_=0.0001)
    except Exception as e:
        error_msg = f"Error in train trlstm:{str(e)}"
        logger.error(error_msg)