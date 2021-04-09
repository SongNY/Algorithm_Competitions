
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DNN(nn.Module):
    """不考虑序列交互的DNN模型
    """

    def __init__(self, w2v_dict, *args, **kwargs):
        super().__init__()
        self.embeds_dict = nn.ModuleDict([])
        for key in w2v_dict:
            w2v = w2v_dict[key]
            # w2v的值有两个可能，如果是一个numpy数组，则是预训练好的word2vec
            # 如果是两个有两个元素的tuple，则第一个元素表示词典大小，第二个元素表示维度
            if isinstance(w2v, np.ndarray):
                w2v = torch.from_numpy(w2v)
                self.embeds_dict[key] = nn.Embedding.from_pretrained(w2v, padding_idx=0)
            else:
                self.embeds_dict[key] = nn.Embedding(w2v[0], w2v[1], padding_idx=0)  # 使用[-0.2, 0.2]的均匀分布初始化
                self.embeds_dict[key].weight.data.uniform_(-0.2, 0.2)
        self.ader_dense = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU()
        )
        self.item_dense = nn.Sequential(
            nn.Linear(512, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 2048),
            nn.LeakyReLU()
        )
        self.seq_dense = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 8192),
            nn.LeakyReLU(),
            nn.Linear(8192, 20)
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        device = self.device

        ader_ids = inputs["ader"].to(device)
        # 有的模型会使用素材id，有的模型会使用广告id，这里做好判断
        if "item" in inputs:
            item_key = "item"
        else:
            item_key = "adid"
        item_ids = inputs[item_key].to(device)

        # 同一个batch中的样本已经按序列长度从长到短排序
        # 这里将同一个batch中的样本划分为几次分别跑，这样可以让长度较短的序列少用一些时间
        # 从而加速训练和预测过程
        # batchsize为512的话会被拆为128+128+256
        dense_input_list = []
        max_len = ader_ids.size(1)  # 整个batch中的最大长度
        batch_size = ader_ids.size(0)
        split_sizes = [batch_size // 4, batch_size // 4, batch_size // 2]
        ader_ids_list = torch.split_with_sizes(ader_ids, split_sizes)
        item_ids_list = torch.split_with_sizes(item_ids, split_sizes)
        for ader_ids0, item_ids0 in zip(ader_ids_list, item_ids_list):
            mask0 = ader_ids0 == 0  # pad的词用0表示
            seq_len0 = max_len - mask0.float().sum(1).long()
            max_len0 = int(seq_len0.max().item())

            # 只截取非pad的部分
            ader_ids0 = ader_ids0[:, :max_len0]
            item_ids0 = item_ids0[:, :max_len0]
            # ader和item的序列分别接一个dense
            ader_embeds = self.embeds_dict["ader"](ader_ids0).float()
            ader_embeds = self.ader_dense(ader_embeds)
            item_embeds = self.embeds_dict[item_key](item_ids0).float()
            item_embeds = self.item_dense(item_embeds)

            # 拼接之后的再经过一个dense
            embeds = torch.cat((ader_embeds, item_embeds), 2)
            embeds = self.seq_dense(embeds)

            # 取maxpooling前将padding位置替换为-1e9
            mask0 = (ader_ids0 == 0).unsqueeze(2).repeat(1, 1, 4096)
            embeds = embeds.masked_fill(mask0, -1e9)
            dense_input_list.append(embeds.max(1)[0])
        dense_input = torch.cat(dense_input_list, 0)
        dense_output = self.dense(dense_input) * 2
        # 如果没输入label，则直接返回dense输出结果
        # 如果输入了label，则计算loss，和dense输出结果一起返回
        if labels is None:
            return dense_output
        labels = labels.to(device)
        loss = self.ce(dense_output, labels)
        return dense_output, loss

    @property
    def device(self):
        # 模型的参数所在的device
        return next(self.parameters()).data.device


class BaseModel(nn.Module):
    """双层lstm和单层transformer+单层transformer的模型定义
    两者很类似，所以在同一个类里定义好
    """

    def __init__(self, model_type, w2v_dict, trainable=False):
        super().__init__()
        d_model = 0
        self.embeds_dict = nn.ModuleDict([])
        self.embeds_linear_dict = nn.ModuleDict([])
        for key in w2v_dict:
            w2v = w2v_dict[key]
            # w2v的值有两个可能，如果是一个numpy数组，则是预训练好的word2vec
            # 如果是两个有两个元素的tuple，则第一个元素表示词典大小，第二个元素表示维度
            if isinstance(w2v, np.ndarray):
                w2v = torch.from_numpy(w2v)
                if trainable and key in ["ader", "prod"]:
                    # 将广告主id和商品id的词向量设成trainable的
                    self.embeds_dict[key] = nn.Embedding.from_pretrained(w2v, padding_idx=0, freeze=False)
                else:
                    self.embeds_dict[key] = nn.Embedding.from_pretrained(w2v, padding_idx=0)
            else:
                self.embeds_dict[key] = nn.Embedding(w2v[0], w2v[1], padding_idx=0)
                self.embeds_dict[key].weight.data.uniform_(-0.2, 0.2)  # 使用[-0.2, 0.2]的均匀分布初始化
            d_model += self.embeds_dict[key].weight.shape[1]
        self.encoder = None
        self.lstm = None
        if model_type == "lstm":
            # 双层双向lstm
            self.lstm = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True, num_layers=2)
        elif model_type == "trlstm":
            # 单层transformer+单层lstm
            encoder_layer = nn.TransformerEncoderLayer(d_model, 2, d_model * 4)
            self.encoder = nn.TransformerEncoder(encoder_layer, 1)
            self.lstm = nn.LSTM(d_model, d_model, bidirectional=True, batch_first=True, num_layers=1)
        self.d_model = d_model
        self.dense_dim = d_model * 2
        self.dropout = nn.Dropout(0.25)
        self.dense = nn.Sequential(
            nn.Linear(self.dense_dim, self.dense_dim * 2),
            nn.GELU(),
            nn.Linear(self.dense_dim * 2, 20)  # 20分类
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        device = self.device
        seq_input_list = []
        # 获取各序列的embedding并拼接
        for key in inputs:
            if key not in self.embeds_dict:
                continue
            ids = inputs[key].to(device)
            embeds = self.embeds_dict[key]
            seq_input_list.append(embeds(ids))
            if embeds.weight.requires_grad:
                # 如果embedding是trainable的，加dropout
                seq_input_list[-1] = self.dropout(seq_input_list[-1])
            else:
                seq_input_list[-1] = seq_input_list[-1].float()
        seq_input = torch.cat(seq_input_list, 2)

        # 同一个batch中的样本已经按序列长度从长到短排序
        # 这里将同一个batch中的样本划分为几次分别跑，这样可以让长度较短的序列少用一些时间
        # 从而加速训练和预测过程
        # batchsize为512的话会被拆为128+128+256
        dense_input_list = []
        max_len = ids.size(1)
        batch_size = ids.size(0)
        mask = ids == 0  # pad的词用0表示
        split_sizes = [batch_size // 4, batch_size // 4, batch_size // 2]
        seq_input_list = torch.split_with_sizes(seq_input, split_sizes)
        mask_list = torch.split_with_sizes(mask, split_sizes)
        for seq_input0, mask0 in zip(seq_input_list, mask_list):
            seq_len0 = max_len - mask0.float().sum(1).long()
            max_len0 = int(seq_len0.max().item())
            # 只截取非pad的部分
            seq_input0 = seq_input0[:, :max_len0, :]
            if self.encoder is not None:
                # 如果encoder不是None，说明是单层transformer+单层lstm的模型
                # 先跑transformer，再将seq_input0设成transformer的输出
                mask0 = mask0[:, :max_len0]
                seq_input0 = self.encoder(
                    seq_input0.permute(1, 0, 2),
                    src_key_padding_mask=mask0
                ).permute(1, 0, 2)
            packed_input = pack_padded_sequence(seq_input0, seq_len0, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_output = pad_packed_sequence(packed_output, batch_first=True, padding_value=-1)[0]
            dense_input_list.append(lstm_output.max(1)[0])

        dense_input = torch.cat(dense_input_list, 0)
        dense_output = self.dense(dense_input)
        # 如果没输入label，则直接返回dense输出结果
        # 如果输入了label，则计算loss，和dense输出结果一起返回
        if labels is None:
            return dense_output
        labels = labels.to(device)
        loss = self.ce(dense_output, labels)
        return dense_output, loss

    @property
    def device(self):
        # 模型的参数所在的device
        return next(self.parameters()).data.device


class BiLSTM(BaseModel):
    """双层lstm
    """

    def __init__(self, *args, **kwargs):
        super().__init__("lstm", *args, **kwargs)


class TransformerWithLSTM(BaseModel):
    """单层transformer+单层lstm
    """

    def __init__(self, *args, **kwargs):
        super().__init__("trlstm", *args, **kwargs)


class WarmupLinearSchedule(torch.optim.lr_scheduler.LambdaLR):
    """一个scheduler，先warmup，再线性递减到0
    从旧版的transformers包中复制出来的
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))
