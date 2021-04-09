
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from tqdm.auto import tqdm

from dataset import TaacDataset
from modeling import BiLSTM, TransformerWithLSTM, DNN, WarmupLinearSchedule
from taac_var import (
    logger, path_cache,
    path_traindata_semifinal, path_traindata_preliminary
)


path_user_dict = path_cache / "user_dict"
path_w2v = path_cache / "word2vec"
path_output = path_cache / f"output"
path_output.mkdir(exist_ok=True, parents=True)


def get_info_dict(setting_list, model_type, ad_map, half=False, trainable=False):
    """读取词向量和词到id的映射
    setting_list: word2vec设定的list，list里面每个元素为一组设定，
        每组设定包含四个元素，分别为：特征名称、维度、窗口长度、低频阈值
    model_type: 模型类型，lstm/trlstm/dnn
    ad_map: 素材id到各特征的映射
    half: 是否在模型中使用半精度词向量，显存足够的话可以忽略
    trainable: 广告主id和商品id是否是可训练的，如果是的话，即使设了half=True，
        对于广告主id和商品id的词向量仍然不会设为半精度
    """
    word2index_dict = {}
    embeds_dict = {}

    for setting in setting_list:
        # 每组设定包含四个元素，分别为：特征名称、维度、窗口长度、低频阈值
        key, dim, window, min_count = setting
        name = f"{key}{dim}/win{window}_min{min_count}"
        with open(path_w2v / (name + ".txt"), "r") as f:
            index2word = f.read().strip().split("\n")
            index2word[2:] = [int(x) for x in index2word[2:]]  # 词使用整数
            word2index_dict[key] = dict(zip(index2word, range(len(index2word))))
        embeds_dict[key] = np.load(path_w2v / (name + ".npy"))
        # 第一个词是pad，使用随机向量作为其向量
        # 第二个词是unk，将其他所有词向量均值作为其向量
        embeds_dict[key][0, :] = np.random.uniform(-0.1, 0.1, size=(embeds_dict[key].shape[1], ))
        embeds_dict[key][1, :] = embeds_dict[key].mean(0)
        if half and (not trainable or key not in ["ader", "prod"]):
            embeds_dict[key] = embeds_dict[key].astype(np.float16)

    if model_type != "dnn":
        # 非dnn模型中使用商品类型和行业id的序列，使用的embedding维度都是64
        cate_set = set(ad_map["cate"].values())
        word2index_dict["cate"] = dict(zip(cate_set, range(2, 2 + len(cate_set))))
        word2index_dict["cate"].update({"pad": 0, "null": 1})
        embeds_dict["cate"] = (len(cate_set) + 2, 64)

        ind_set = set(ad_map["ind"].values())
        word2index_dict["ind"] = dict(zip(ind_set, range(2, 2 + len(ind_set))))
        word2index_dict["ind"].update({"pad": 0, "null": 1})
        embeds_dict["ind"] = (len(ind_set) + 2, 64)

    return word2index_dict, embeds_dict


def sort_batch(batch):
    """将batch中的样本按序列长度从长到短排序，方便之后做一个加速操作
    """
    batch = default_collate(batch)
    inputs, labels = batch
    ids = list(inputs.values())[0]

    mask = ids == 0
    max_len = ids.size(1)
    seq_len = max_len - mask.float().sum(1).long()

    sort_index = sorted(range(len(seq_len)), key=lambda k: -seq_len[k])
    inputs = inputs.copy()
    for key in inputs:
        inputs[key] = inputs[key][sort_index]
    labels = labels[sort_index]
    return inputs, labels


model_setting_dict = {
    # 模型设定
    # model_type: lstm或者trlstm或者dnn
    # window_list: 要用哪些window大小的词向量，list的长度也等于这组模型的训练个数
    # item_key_list: 有些模型用素材id序列，有些模型用广告id序列，这里决定每个模型用哪个
    # w2v_dim: 每组特征的维度
    # min_count: 词向量的min_count
    # trainable: 是否把用户id和商品id的词向量设成可训练的
    "lstm1": {
        "model_type": "lstm",
        "window_list": [2 ** x for x in range(2, 8)],
        "item_key_list": ["item", "adid"] * 3,
        "w2v_dim": {"ader": 384, "item": 256, "adid": 256, "prod": 128},
        "min_count": 1,
        "trainable": False
    },
    "trlstm1": {
        "model_type": "trlstm",
        "window_list": [2 ** x for x in range(2, 8)],
        "item_key_list": ["adid", "item"] * 3,
        "w2v_dim": {"ader": 384, "item": 256, "adid": 256, "prod": 128},
        "min_count": 1,
        "trainable": False
    },
    "trlstm2": {
        "model_type": "trlstm",
        "window_list": [2 ** x for x in range(2, 7)],
        "item_key_list": ["adid", "item"] * 3,
        "w2v_dim": {"ader": 384, "item": 256, "adid": 256, "prod": 128},
        "min_count": 1,
        "trainable": True
    },
    "dnn1": {
        "model_type": "dnn",
        "window_list": [2 ** x for x in range(2, 8)],
        "item_key_list": ["adid", "item"] * 3,
        "w2v_dim": {"ader": 512, "item": 512, "adid": 512},
        "min_count": 1,
        "trainable": False
    },
    "dnn2": {
        "model_type": "dnn",
        "window_list": [2 ** x for x in range(2, 8)],
        "item_key_list": ["adid", "item"] * 3,
        "w2v_dim": {"ader": 512, "item": 512, "adid": 512},
        "min_count": 4,
        "trainable": False
    }
}

# 各个模型类型的参数
lr_dict = {"lstm": 2e-3, "trlstm": 3e-4, "dnn": 2e-4}  # 学习率
warmup_dict = {"lstm": 0.01, "trlstm": 0.1, "dnn": 0.05}  # 使用多大比率的step来做warmup
cuda_dict = {"lstm": 0, "trlstm": 0, "dnn": 0}  # 使用的cuda设备
num_epochs_dict = {"lstm": 4, "trlstm": 4, "dnn": 5}  # 训练的epoch数量（比线下最优epoch数多一个）
model_class_dict = {"lstm": BiLSTM, "trlstm": TransformerWithLSTM, "dnn": DNN}  # 模型的类
trunc_mode_list = ["random", "left", "middle", "right", "random", "random"]  # 序列截断类型的list


if __name__ == "__main__":
    model_key = sys.argv[1]
    logger.info(f"开始模型{model_key}的训练")
    model_setting = model_setting_dict[model_key]
    model_type = model_setting["model_type"]
    lr = lr_dict[model_type]
    warmup_prop = warmup_dict[model_type]
    cuda_id = cuda_dict[model_type]
    num_epochs = num_epochs_dict[model_type]
    model_class = model_class_dict[model_type]
    if len(sys.argv) > 2:
        cuda_id = int(sys.argv[2])

    logger.info(f"载入模型{model_key}的数据")
    # 读取用户label
    user_df1 = pd.read_csv(path_traindata_preliminary / "user.csv")
    user_df2 = pd.read_csv(path_traindata_semifinal / "user.csv")
    user_df = pd.concat([user_df1, user_df2])
    user_labels = user_df["gender"] * 10 + user_df["age"] - 11  # 20分类
    user_label_map = dict(zip(user_df["user_id"], user_labels))

    # 用户点击的广告序列和相应的时间序列
    user_seq_dict = {}
    with open(path_user_dict / "item.pkl", "rb") as f:
        user_seq_dict["item"] = pickle.load(f)
    with open(path_user_dict / "time.pkl", "rb") as f:
        user_seq_dict["time"] = pickle.load(f)
    with open(path_cache / "ad_map.pkl", "rb") as f:
        ad_map = pickle.load(f)

    for i in range(len(model_setting["window_list"])):
        logger.info(f"开始模型{model_key}的第{i + 1}个模型")
        trainable = model_setting["trainable"]
        min_count = model_setting["min_count"]
        window = model_setting["window_list"][i]
        item_key = model_setting["item_key_list"][i]
        w2v_dim_dict = model_setting["w2v_dim"].copy()
        # 如果是使用素材id的词向量，则删除广告id的词向量设定，反之亦然
        if item_key == "item":
            del w2v_dim_dict["adid"]
        elif item_key == "adid":
            del w2v_dim_dict["item"]

        # word2vec设定的list，list里面每个元素为一组设定
        # 每组设定包含四个元素，分别为：特征名称、维度、窗口长度、低频阈值
        w2v_setting_list = []
        for key, val in w2v_dim_dict.items():
            w2v_setting_list.append((key, val, window, min_count))

        # 读取词向量和词到id的映射
        word2index_dict, embeds_dict = get_info_dict(w2v_setting_list, model_type, ad_map)
        seed = int(time.time() % 1e5 * 1000)
        trunc_mode = trunc_mode_list[i]

        # 训练和测试集数据集及dataloader，使用的batch_size为512
        train_dataset = TaacDataset(
            user_seq_dict, ad_map, word2index_dict, user_label_map, "train",
            max_len=128, trunc_mode=trunc_mode
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=512, shuffle=True,
            num_workers=4, collate_fn=sort_batch
        )
        test_dataset = TaacDataset(
            user_seq_dict, ad_map, word2index_dict, None, "test",
            max_len=128, trunc_mode=trunc_mode
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=512,
            num_workers=4, collate_fn=sort_batch
        )

        m = model_class(embeds_dict, trainable).cuda(cuda_id)
        total_steps = len(train_dataloader) * num_epochs  # 总步数
        warmup_steps = int(total_steps * warmup_prop)  # warmup的步数
        optimizer = torch.optim.AdamW(m.parameters(), lr)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps, total_steps)  # 先warmup，然后线性递减到0

        logger.info(f"模型{model_key}的第{i + 1}个模型准备完成，开始训练")
        # 训练
        for epoch in range(1, num_epochs + 1):
            t = tqdm(train_dataloader)
            for inputs, labels in t:
                m.train()
                output, loss = m(inputs, labels)
                loss.backward()
                t.set_postfix(loss=loss.item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            logger.info(f"模型{model_key}的第{i + 1}个模型完成第{epoch}个epoch")

        logger.info(f"模型{model_key}的第{i + 1}个模型训练完成，开始预测")
        # 预测
        all_output_list = []
        t = tqdm(test_dataloader)
        for inputs, user_ids in t:
            m.eval()
            sort_index = sorted(range(len(user_ids)), key=user_ids.__getitem__)
            with torch.no_grad():
                # 同一个batch中的结果重新按用户id大小排序
                output = m(inputs)[sort_index]
            all_output_list.append(output.softmax(-1).cpu())
        all_output = torch.cat(all_output_list, 0).numpy()
        # 保存模型输出结果
        logger.info(f"模型{model_key}的第{i + 1}个模型预测完成，保存预测概率")
        torch.save(all_output, path_output / f"output_{model_key}_{i + 1}")
