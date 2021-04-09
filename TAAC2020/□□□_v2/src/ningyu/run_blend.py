# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import numpy as np
import os
from taacalgo.util import path_tmpdata
from taacalgo.util import logger

path_output = path_tmpdata / "output/"
files = os.listdir(path_output)
all_pred = np.empty([1000000,20])
for filename in files:
    all_pred = all_pred + np.load(path_output / filename)
    
np.save(path_tmpdata / "model24.npy", all_pred)
# log
logger.info('successful output model24.npy')