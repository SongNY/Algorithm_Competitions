# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/11/18

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
logger.info('successful train w2v')

# get pad_clicktime
behavior_time=click.groupby('user_id')['click_times'].apply(lambda x:x.values.tolist())
get_pad_clicktime(behavior_time)

# log
logger.info('successful train w2v')