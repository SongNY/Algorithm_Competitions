# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pad_packed_sequence
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
from .util import path_tmpdata
from .util import logger

def set_seed(seed):
    """ 设置随机数种子
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compute_acc(y1, y2):
    """ 计算准确率
    :return: 准确率
    """
    return accuracy_score(y1.cpu().numpy(), y2.sigmoid().detach().cpu().numpy().argmax(axis=1))

# 单层transformer + 双层lstm
class tr_lstm(nn.Module):
    def __init__(self, hidden_dim, w2v_dim, out_dim , freeze_adver ,embed_ct, embed_adver, embed_clicktime):
        super(tr_lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v_dim = w2v_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(self.w2v_dim * 2, self.hidden_dim, 2, batch_first=True, bidirectional=True)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=4),
                                                         num_layers=1)
        self.embed_ct = nn.Embedding.from_pretrained(embed_ct)
        self.embed_adver = nn.Embedding.from_pretrained(embed_adver, freeze=freeze_adver)
        self.embed_clicktime = nn.Embedding.from_pretrained(embed_clicktime, freeze=False)
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 2, self.out_dim)
        )

    def forward(self, ct, adver, clicktime, x_len, mask):
        ct_emb = self.embed_ct(ct).float()
        adver_emb = self.embed_adver(adver).float()
        clicktime_emb = self.embed_clicktime(clicktime).float()
        cat_emb = torch.cat((ct_emb, adver_emb), dim=2)
        cat_emb = cat_emb * clicktime_emb
        tr_out = self.transformer_encoder(cat_emb.permute(1, 0, 2),
                                          src_key_padding_mask=mask)
        cat_pack = pack_padded_sequence(tr_out.permute(1, 0, 2), x_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(cat_pack)
        output = pad_packed_sequence(output, batch_first=True, padding_value=-1)
        output1 = output[0].max(1)[0].view(-1, self.hidden_dim * 2)
        out = self.dense(output1)
        return out

# 单层transformer + 双层lstm + DNN
class tr_lstm_dnn(nn.Module):
    def __init__(self, hidden_dim, w2v_dim, out_dim, embed_ct, embed_adver, embed_clicktime):
        super(tr_lstm_dnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.w2v_dim = w2v_dim
        self.out_dim = out_dim
        self.lstm = nn.LSTM(self.w2v_dim * 2, self.hidden_dim, 2, batch_first=True, bidirectional=True)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=4),
                                                         num_layers=1)
        self.embed_ct = nn.Embedding.from_pretrained(embed_ct)
        self.embed_adver = nn.Embedding.from_pretrained(embed_adver, freeze=False)
        self.embed_clicktime = nn.Embedding.from_pretrained(embed_clicktime, freeze=False)
        self.dense = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 8),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim * 2, self.out_dim)
        )

    def forward(self, ct, adver, clicktime, x_len, mask):
        ct_emb = self.embed_ct(ct).float()
        adver_emb = self.embed_adver(adver).float()
        clicktime_emb = self.embed_clicktime(clicktime).float()
        cat_emb = torch.cat((ct_emb, adver_emb), dim=2)
        cat_emb = cat_emb * clicktime_emb
        tr_out = self.transformer_encoder(cat_emb.permute(1, 0, 2),
                                          src_key_padding_mask=mask)
        cat_pack = pack_padded_sequence(tr_out.permute(1, 0, 2), x_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(cat_pack)
        output = pad_packed_sequence(output, batch_first=True, padding_value=-1)
        output1 = output[0].max(1)[0].view(-1, self.hidden_dim * 2)
        out = self.dense(output1)
        return out

# 双层transformer
class tr(nn.Module):
    def __init__(self,hidden_dim, out_dim, embed_ct, embed_adver, embed_clicktime):
        super(tr, self).__init__()
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8),
                                                         num_layers=2)
        self.embed_ct = nn.Embedding.from_pretrained(embed_ct)
        self.embed_adver = nn.Embedding.from_pretrained(embed_adver)
        self.embed_clicktime = nn.Embedding.from_pretrained(embed_clicktime,freeze=False)
        self.dense =  nn.Sequential(
            nn.Linear(self.hidden_dim*4,self.hidden_dim*4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim*4,self.out_dim)
        )
        self.dense_tr =  nn.Sequential(
            nn.Linear(512,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,2048)
        )

    def forward(self,ct,adver,clicktime,x_len,mask):
        ct_emb = self.embed_ct(ct).float()
        adver_emb = self.embed_adver(adver).float()
        clicktime_emb = self.embed_clicktime(clicktime).float()
        cat_emb = torch.cat((ct_emb,adver_emb),dim=2)
        cat_emb = cat_emb * clicktime_emb
        tr_out = self.transformer_encoder(cat_emb.permute(1,0,2),
                                          src_key_padding_mask=mask)
        tr_out = self.dense_tr(tr_out.permute(1,0,2))
        out_pack = pack_padded_sequence(tr_out,x_len,batch_first=True,enforce_sorted=False)
        out_pad = pad_packed_sequence(out_pack,batch_first=True,padding_value=-1,total_length=128)
        output = out_pad[0].max(1)[0].view(-1,self.hidden_dim*4)
        out = self.dense(output)
        return  out

# 双层transformer 快速版
class tr_fast(nn.Module):
    def __init__(self,hidden_dim, out_dim, embed_ct, embed_adver, embed_clicktime):
        super(tr_fast, self).__init__()
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8),
                                                         num_layers=2)
        self.embed_ct = nn.Embedding.from_pretrained(embed_ct*0.9)
        self.embed_adver = nn.Embedding.from_pretrained(embed_adver*1.1)
        self.embed_clicktime = nn.Embedding.from_pretrained(embed_clicktime,freeze=False)
        self.dense =  nn.Sequential(
            nn.Linear(self.hidden_dim*4,self.hidden_dim*4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim*4,self.out_dim)
        )
        self.dense_tr =  nn.Sequential(
            nn.Linear(512,2048),
            nn.LeakyReLU(),
            nn.Linear(2048,2048)
        )

    def forward(self,ct,adver,clicktime,x_len,mask):
        ct_emb = self.embed_ct(ct).float()
        adver_emb = self.embed_adver(adver).float()
        clicktime_emb = self.embed_clicktime(clicktime).float()
        cat_emb = torch.cat((ct_emb,adver_emb),dim=2)
        cat_emb = cat_emb * clicktime_emb
        tr_out = self.transformer_encoder(cat_emb.permute(1,0,2),
                                          src_key_padding_mask=mask)
        out_pack = pack_padded_sequence(tr_out.permute(1,0,2),x_len,batch_first=True,enforce_sorted=False)
        out_pad = pad_packed_sequence(out_pack,batch_first=True,padding_value=0,total_length=128)
        output = self.dense_tr(out_pad[0])
        output = output.max(1)[0].view(-1,self.hidden_dim*4)
        out = self.dense(output)
        return  out


def train_model(train_user, models, win, plusminus, seeddd, epoch_=4, lr_=0.004, freeze_adver=True):

    """ train model & save model and output

    :param train_user y
    :param models: 训练所用model名称
    :param win: 窗口大小
    :param plusminus: 正负序列
    :param seeddd: 种子
    :param epoch_: 训练轮数
    :param lr_: 最大学习率
    :param freeze_adver: embedding_advertiser 是否不可训练

    :return:

    """

    # log
    logger.info('start train model:%s windows:%d sequence:%s128 seed:%d' % (models, win, plusminus, seeddd))

    # 导入数据
    path_pad_ct_name = 'pad_creative_' + plusminus + '128'
    path_embed_ct_name = 'embed_creative_w'+str(win)
    pad_ct=torch.load(path_tmpdata / path_pad_ct_name)
    embed_ct=torch.load(path_tmpdata / path_embed_ct_name)

    path_pad_adver_name = 'pad_advertiser_' + plusminus + '128'
    path_embed_adver_name = 'embed_advertiser_w'+str(win)
    pad_adver=torch.load(path_tmpdata / path_pad_adver_name)
    embed_adver=torch.load(path_tmpdata / path_embed_adver_name)

    path_pad_len_name = 'pad_len_'+plusminus+'128'
    path_pad_clicktime_name = 'pad_clicktime_'+plusminus+'128'
    pad_clicktime=torch.load(path_tmpdata / path_pad_clicktime_name)
    len_pad=torch.load(path_tmpdata / path_pad_len_name)

    # clicktime截断并取ln
    pad_clicktime[pad_clicktime > 10] = 10
    embed_clicktime = torch.zeros(1, 512)
    for i in range(10):
        embed_clicktime = torch.cat((embed_clicktime, math.log(i + math.e) * torch.ones(1, 512)), dim=0)

    # 设置随机数种子
    set_seed(seeddd)

    # 构造训练集
    dataset = TensorDataset(pad_ct[:3000000], pad_adver[:3000000],
                            pad_clicktime[:3000000], len_pad[:3000000],
                            torch.from_numpy(train_user['label'].values))
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    # 确定device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型选取
    if models=='tr':
        model_lstm = tr(512, 20, embed_ct,embed_adver, embed_clicktime).to(device)
    if models=='tr2':
        model_lstm = tr_fast(512, 20, embed_ct,embed_adver, embed_clicktime).to(device)
    if models=='trlstm':
        model_lstm = tr_lstm(1024, 256, 20, freeze_adver,embed_ct,embed_adver, embed_clicktime).to(device)
    if models=='trlstmdnn':
        model_lstm = tr_lstm_dnn(1024, 256, 20, embed_ct,embed_adver, embed_clicktime).to(device)

    # 损失函数
    criterion = F.cross_entropy

    # 优化器
    optimizer = torch.optim.AdamW(model_lstm.parameters(), lr=lr_)

    # train
    for epoch in range(epoch_):
        model_lstm.train()
        acc0 = []
        for ct, adver, c_time, x_len, target in tqdm(dataloader):
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_ * ((i + 1) / 3000)
            if epoch == epoch_-1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_ * (1 - i / 3000)

            mask = torch.ones(len(x_len), 128)
            for i in range(len(x_len)):
                mask[i, :x_len[i]] = 0
            mask = mask.bool()

            output = model_lstm(ct.to(device), adver.to(device), c_time.to(device), x_len, mask.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            accc = compute_acc(target, output)
            acc0.append(accc)
            optimizer.step()
            optimizer.zero_grad()

        # log
        logger.info('train_acc : %f' % pd.DataFrame(acc0).mean().values)

    # save model
    save_model_name = 'model/'+models+'_s'+str(seeddd)+'_w'+str(win)+plusminus
    torch.save(model_lstm.state_dict(), path_tmpdata / save_model_name)

    # eval
    model_lstm.eval()
    m = torch.nn.Softmax(dim=1)
    test_dataset = TensorDataset(pad_ct[3000000:], pad_adver[3000000:],
                                 pad_clicktime[3000000:], len_pad[3000000:])
    test_dataloader = DataLoader(test_dataset, batch_size=2000, shuffle=False)
    all_pred = np.empty([0, 20])
    for ct, adver, c_time, x_len in tqdm(test_dataloader):
        with torch.no_grad():
            mask = torch.ones(len(x_len), 128)
            for i in range(len(x_len)):
                mask[i, :x_len[i]] = 0
            mask = mask.bool()
            output = model_lstm(ct.to(device), adver.to(device), c_time.to(device),
                                x_len, mask.to(device))
            all_output = m(output).detach().cpu().numpy()
            all_pred = np.append(all_pred, all_output, axis=0)

    # 非过拟合 概率减半
    if epoch_==3:
        all_pred=0.5*all_pred
    if (epoch_==4) & ((models=='tr1') | (models=='tr2')):
        all_pred=0.5*all_pred

    # save output
    save_output_name = 'output/'+models+'_s'+str(seeddd)+'_w'+str(win)+plusminus+'.npy'
    np.save(path_tmpdata / save_output_name, all_pred)

    # 清理cuda
    model_lstm.to('cpu')
    del model_lstm
    torch.cuda.empty_cache()
