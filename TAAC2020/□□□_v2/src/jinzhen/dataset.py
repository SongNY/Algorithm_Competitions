
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class TaacDataset(Dataset):
    """模型训练时使用的dataset
    """
    train_user_range = range(1, 3000001)  # 训练集用户id范围
    test_user_range = range(3000001, 4000001)  # 测试集用户id范围

    def __init__(
        self, user_seq_dict, ad_map, word2index_dict, user_label_map,
        data_type, max_len=96, trunc_mode="right", seed=None,
        split_eval_size=0.05, split_seed=233
    ):
        assert trunc_mode in ["left", "right", "middle", "random"]
        assert data_type in ["train", "test", "split_train", "split_eval", "all"]
        self.user_seq_dict = user_seq_dict  # 用户序列的dict
        self.ad_map = ad_map  # 素材id到各种属性的映射
        self.word2index_dict = word2index_dict  # 词到词id的映射
        self.user_label_map = user_label_map  # 用户id到label（20分类）的映射
        self.data_type = data_type  # 数据类型：训练集、测试集……
        self.max_len = max_len  # 最大序列长度
        self.trunc_mode = trunc_mode  # 截断类型：截左边、截右边、截中间、随机截
        self.seed = seed  # 生成序列的随机数种子
        self.split_eval_size = split_eval_size  # 训练集划分出来的验证集比例
        self.split_seed = split_seed  # 用于划分训练集和验证集的种子
        if self.seed is None:
            self.seed = int(time.time() * 1000 % 1e8)
        if data_type == "train":
            # 全量训练集
            self.user_list = list(self.train_user_range)
        elif data_type == "test":
            # 测试集
            self.user_list = list(self.test_user_range)
        elif data_type == "all":
            # 全量训练集和测试集一起用
            self.user_list = list(self.train_user_range) + list(self.test_user_range)
        else:
            # 将训练集划分为线下评测用的训练集和验证集
            split_res = train_test_split(
                self.train_user_range,
                test_size=split_eval_size,
                random_state=split_seed
            )
            if data_type == "split_train":
                self.user_list = split_res[0]
            else:
                self.user_list = split_res[1]

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        user_id = self.user_list[index]
        max_len = self.max_len
        # 每个用户在同一个模型中的不同epoch使用相同种子
        rng = np.random.RandomState(self.seed + user_id)

        item_seq = self.user_seq_dict["item"][user_id]
        time_seq = self.user_seq_dict["time"][user_id]

        # 同一个日期内的广告进行shuffle
        seq_len = len(item_seq)
        resort_index_list = list(range(seq_len))
        np.random.shuffle(resort_index_list)
        resort_index_list = sorted(resort_index_list, key=time_seq.__getitem__)

        item_seq = [item_seq[i] for i in resort_index_list]
        time_seq = [time_seq[i] for i in resort_index_list]

        # 序列长度太长时的截断
        if seq_len > max_len:
            if self.trunc_mode == "right":
                # 保留最右的一整段
                # 随机偏移一个位置
                if rng.random() < 0.5:
                    s = slice(-max_len - 1, -1)
                else:
                    s = slice(-max_len, None)
            elif self.trunc_mode == "left":
                # 保留最左的一整段
                # 随机偏移一个位置
                if rng.random() < 0.5:
                    s = slice(1, max_len + 1)
                else:
                    s = slice(None, max_len)
            elif self.trunc_mode == "middle":
                # 保留正中间的一整段
                # 随机偏移一个位置
                left = (seq_len - max_len) // 2
                right = left + max_len
                if rng.random() < 0.5:
                    s = slice(left, right)
                else:
                    s = slice(left + 1, right + 1)
            elif self.trunc_mode == "random":
                # 随机从序列中抽取，但仍保留顺序
                s = np.random.choice(range(seq_len), max_len, replace=False)
                s = sorted(s)
            if isinstance(s, slice):
                item_seq = item_seq[s]
                time_seq = time_seq[s]
            else:
                item_seq = [item_seq[i] for i in s]
                time_seq = [time_seq[i] for i in s]
        res = {}
        pad_len = max_len - seq_len
        for key in self.word2index_dict:
            if key == "item":
                seq = item_seq
            elif key == "time":
                seq = time_seq
            else:
                seq = [self.ad_map[key][x] for x in item_seq]
            # 序列转换，如果词典中找不到该词，就替换为1 （UNK所对应的index）
            # pad的词用0表示
            seq = [self.word2index_dict[key].get(x, 1) for x in seq] + [0] * pad_len
            res[key] = torch.LongTensor(seq)
        # 如果有label的话则返回label，没有的话则返回用户id
        if self.user_label_map is not None and user_id in self.user_label_map:
            label = self.user_label_map[user_id]
            return res, label
        return res, user_id
