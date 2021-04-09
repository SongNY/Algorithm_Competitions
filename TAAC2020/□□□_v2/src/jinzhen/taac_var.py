
import logging
from pathlib import Path

# log, cache和rawdata等文件夹和src文件夹是同一级
path_log = Path("../../log/")
path_cache = Path("../../cache/")
path_rawdata = Path("../../rawdata/")
path_traindata_preliminary = path_rawdata / "train/preliminary"
path_traindata_semifinal = path_rawdata / "train/semi_final"
path_testdata = path_rawdata / "test"

all_vars = globals().copy()
for name, var in all_vars.items():
    if name.startswith("path_") and isinstance(var, Path):
        var.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("jinzhen_taac")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(path_log / "jinzhen_taac.log", "a")
formatter = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
