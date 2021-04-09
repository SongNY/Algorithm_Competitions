# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import pandas
from taacalgo.dao import w2v_dao
from taacalgo.w2v import train_w2v, train_w2v_get_pad, get_pad_clicktime
from taacalgo.util import logger

# train creative_id w2v
## groupby user_id get creative_id's behavior_list
click=w2v_dao()
# log
logger.info('successful input click')
click['creative_id']=click['creative_id'].astype(str)
behavior_list=click.groupby('user_id')['creative_id'].apply(lambda x:x.values.tolist())

## train_w2v and get pad
# log
logger.info('start train creative w2v')
train_w2v_get_pad(behavior_list, adtype='creative', win=16)
## train_w2v
train_w2v(behavior_list, adtype='creative', win=6, negative=13)
train_w2v(behavior_list, adtype='creative', win=32)
train_w2v(behavior_list, adtype='creative', win=64)

# train advertiser_id w2v
## groupby user_id get advertiser_id's behavior_list
click['advertiser_id']=click['advertiser_id'].astype(str)
behavior_list2=click.groupby('user_id')['advertiser_id'].apply(lambda x:x.values.tolist())

## train_w2v and get pad
# log
logger.info('start train advertiser w2v')
train_w2v_get_pad(behavior_list2, adtype='advertiser', win=16)
## train_w2v
train_w2v(behavior_list2, adtype='advertiser', win=6, negative=13)
train_w2v(behavior_list2, adtype='advertiser', win=32)
train_w2v(behavior_list2, adtype='advertiser', win=64)

# get pad_clicktime
behavior_time=click.groupby('user_id')['click_times'].apply(lambda x:x.values.tolist())
get_pad_clicktime(behavior_time)
# log
logger.info('successful train w2v')