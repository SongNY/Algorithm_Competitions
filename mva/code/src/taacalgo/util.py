# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import logging
from pathlib import Path

path_tmpdata = Path("../../tmp_data/")
path_output = path_tmpdata / "output"
path_model = path_tmpdata / "model"
path_rawdata = Path("../../rawdata/")
path_traindata_prelimnary = path_rawdata / "train/preliminary"
path_traindata_semifinal = path_rawdata / "train/semi_final"
path_testdata = path_rawdata / "test"
path_log = Path("../../log/")

all_vars = globals().copy()
for name, var in all_vars.items():
    if name.startswith("path_") and isinstance(var, Path):
        var.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("ningyu_taac")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(path_log / "ningyu_taac.log", "a")
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)