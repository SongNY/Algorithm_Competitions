
import pickle

import numpy as np

from taac_var import logger, path_cache


def generate_corpus(field_list, seed=0):
    """生成各种序列的“语料”，用于训练word2vec
    将序列存为文本文件，再使用c版本的word2vec进行训练。
    “语料”中每一行对应一个用户，多个词用空格隔开。
    field_list表示要生成哪些序列的“语料”，item表示素材id，adid表示广告id，ader表示广告主id，prod表示商品id
    seed为用来对用户进行shuffle的随机数种子
    """
    logger.info("开始载入序列语料")
    rng = np.random.RandomState(seed=seed)
    path_corpus = path_cache / "corpus"
    path_user_dict = path_cache / "user_dict"
    path_corpus.mkdir(exist_ok=True, parents=True)

    with open(path_user_dict / "item.pkl", "rb") as f:
        user_item_dict = pickle.load(f)
    with open(path_cache / "ad_map.pkl", "rb") as f:
        ad_map = pickle.load(f)

    fo_dict = {}
    for field in field_list:
        fo_dict[field] = open(path_corpus / (field + ".txt"), "w")
    user_item_list = list(user_item_dict.values())
    rng.shuffle(user_item_list)
    for item_seq in user_item_list:
        # item_seq是素材id的序列
        for field in field_list:
            if field == "item":
                seq = item_seq
            else:
                # 对于其他字段，将素材id序列转换过来
                seq = [ad_map[field][x] for x in item_seq]
            f = fo_dict[field]
            f.write(" ".join(str(x) for x in seq) + "\n")
    for f in fo_dict.values():
        f.close()
    logger.info("载入序列语料完成")


if __name__ == "__main__":
    generate_corpus(["ader", "item", "prod", "adid"])
