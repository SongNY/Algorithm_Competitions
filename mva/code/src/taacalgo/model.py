# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/12/6

import time
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

# model
class seq_model(nn.Module):
    def __init__(self, model_name, embed_ct, embed_clicktime, 
                 rnn_layer=0, gru_layer=0, lstm_layer=0, tr_layer=0, tr_head=4, dnn_dim=2048, bidirectt=True):
        super(seq_model, self).__init__()
        self.model_name = model_name
        self.dnn_dim = dnn_dim
        if 'lstm' in self.model_name:
            self.lstm = nn.LSTM(512, 512, lstm_layer, batch_first=True, bidirectional=bidirectt)
        if 'gru' in self.model_name:
            self.gru = nn.GRU(512, 512, gru_layer, batch_first=True, bidirectional=bidirectt)
        if 'rnn' in self.model_name:
            self.rnn = nn.RNN(512, 512, rnn_layer, batch_first=True, bidirectional=bidirectt)
        if 'tr' in self.model_name:
            self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=tr_head),
                                                             num_layers=tr_layer)
        self.embed_ct = nn.Embedding.from_pretrained(embed_ct)
        self.embed_clicktime = nn.Embedding.from_pretrained(embed_clicktime, freeze=False)
        input_dim = 1024 if (('lstm' in self.model_name or 'gru' in self.model_name or 'rnn' in self.model_name) and bidirectt) else 512
        if dnn_dim:
            self.dense1 = nn.Sequential(
                nn.Linear(input_dim, dnn_dim),
                nn.LeakyReLU(),
                nn.Linear(dnn_dim, dnn_dim)
            )
            self.dense2 = nn.Sequential(
                nn.Linear(dnn_dim, dnn_dim),
                nn.LeakyReLU(),
                nn.Linear(dnn_dim, 20)
            )
        else:
            self.dense2 = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LeakyReLU(),
                nn.Linear(input_dim, 20)
            )

    def forward(self, ct, clicktime, x_len, mask):
        ct_emb = self.embed_ct(ct).float()
        clicktime_emb = self.embed_clicktime(clicktime).float()
        out_emb = ct_emb * clicktime_emb
        if 'tr' in self.model_name:
            tr_out = self.transformer_encoder(out_emb.permute(1, 0, 2),
                                              src_key_padding_mask=mask)
            out_emb = tr_out.permute(1, 0, 2)
        out_pack = pack_padded_sequence(out_emb, x_len, batch_first=True, enforce_sorted=False)
        if 'lstm' in self.model_name:
            out_pack, (_, _) = self.lstm(out_pack)
        if 'gru' in self.model_name:
            out_pack, (_, _) = self.gru(out_pack)
        if 'rnn' in self.model_name:
            out_pack, (_, _) = self.rnn(out_pack)
        out_pad = pad_packed_sequence(out_pack, batch_first=True, padding_value=0)
        if self.dnn_dim:
            output = self.dense1(out_pad[0])
        else:
            output = out_pad[0]
        maxpooling_dim = self.dnn_dim if self.dnn_dim else 512
        output_max = output.max(1)[0].view(-1, maxpooling_dim)
        out = self.dense2(output_max)
        return out


def train_model(train_user, models, seeddd, epoch_=4, lr_=0.001,
                rnn_layer=0, gru_layer=0, lstm_layer=0, tr_layer=0, tr_head=4, dnn_dim=2048, bidirectt=True):

    
    """ train model & save model and output

    :param train_user y
    :param models: 训练所用model名称
    :param seeddd: 种子
    :param epoch_: 训练轮数
    :param lr_: 最大学习率

    :return:

    """

    # log
    logger.info('start train model:%s lstm_layer:%d tr_layer:%d tr_head:%d rnn_layer:%d gru_layer:%d dnn_dim:%d seed:%d'
                % (models, lstm_layer, tr_layer, tr_head, rnn_layer, gru_layer, dnn_dim, seeddd))
    logger.info('model setting: epoch:%d lr:%f'
                % (epoch_,lr_))

    # 导入数据
    path_pad_ct_name = 'pad_creative_+128'
    path_embed_ct_name = 'embed_creative_w'+str(16)
    pad_ct=torch.load(path_tmpdata / path_pad_ct_name)
    embed_ct=torch.load(path_tmpdata / path_embed_ct_name)

    path_pad_len_name = 'pad_len_+128'
    path_pad_clicktime_name = 'pad_clicktime_+128'
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
    dataset = TensorDataset(pad_ct,pad_clicktime, len_pad,
                            torch.from_numpy(train_user['label'].values))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [2000000, 1000000])
    dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    # 确定device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 模型选取
    model_lstm = seq_model(models, embed_ct, embed_clicktime, 
                           rnn_layer, gru_layer, lstm_layer, tr_layer, tr_head, dnn_dim, bidirectt).to(device)

    # 损失函数
    criterion = F.cross_entropy

    # 优化器
    optimizer = torch.optim.AdamW(model_lstm.parameters(), lr=lr_)

    # 保存结果
    df_res=[]
    
    # softmax
    softm=torch.nn.Softmax(dim=1)
    
    # train
    for epoch in range(epoch_):
        train_time_start=time.time()
        model_lstm.train()
        acc0 = []
        df_res_tmp={}
        
        for i, (ct, c_time, x_len, target) in enumerate(tqdm(dataloader)):
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_ * ((i + 1) / 20000)
            if epoch == epoch_-1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_ * (1 - i / 20000)

            mask = torch.ones(len(x_len), 128)
            for i in range(len(x_len)):
                mask[i, :x_len[i]] = 0
            mask = mask.bool()

            output = model_lstm(ct.to(device), c_time.to(device), x_len, mask.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            accc = compute_acc(target, output)
            acc0.append(accc)
            optimizer.step()
            optimizer.zero_grad()
        
        train_time_end=time.time()
        # log
        logger.info('train_acc : %f' % pd.DataFrame(acc0).mean().values)
        
        test_time_start=time.time()
        acc_test=[]
        all_pred = np.empty(0)
        all_true = np.empty(0)
        all_pred_prob = np.empty([0,20])
        model_lstm.eval()
        for ct,c_time,x_len,target in tqdm(test_dataloader):
            with torch.no_grad():
                mask = torch.ones(len(x_len),128)
                for i in range(len(x_len)):
                    mask[i,:x_len[i]]=0
                mask = mask.bool()
                output = model_lstm(ct.to(device),c_time.to(device),x_len,mask.to(device))
                accc=compute_acc(target, output)
                acc_test.append(accc)
                all_output = softm(output).detach().cpu().numpy()
                all_pred_prob = np.append(all_output,all_pred_prob,axis=0)
                all_pred = np.append(all_output.argmax(axis=1),all_pred)
                all_true = np.append(target.numpy(),all_true)
        acc_gen=accuracy_score((all_true>9).astype(int),(all_pred>9).astype(int))
        acc_gen2=accuracy_score((all_true>9).astype(int),
                                (all_pred_prob[:,:10].sum(axis=1)<all_pred_prob[:,10:].sum(axis=1)))
        age_pred=(all_pred_prob[:,:10]+all_pred_prob[:,10:]).argmax(axis=1)
        all_true[all_true>9]=all_true[all_true>9]-10
        all_pred[all_pred>9]=all_pred[all_pred>9]-10
        acc_age=accuracy_score(all_true,all_pred)
        acc_age2=accuracy_score(all_true,age_pred)
        logger.info('sum_acc : %f | age_acc : %f | gen_acc : %f | sum_acc : %f' 
          %(pd.DataFrame(acc_test).mean().values,acc_age,acc_gen,acc_age+acc_gen))
        logger.info('sum_acc : %f | age_acc : %f | gen_acc : %f | sum_acc : %f' 
          %(pd.DataFrame(acc_test).mean().values,acc_age2,acc_gen2,acc_age2+acc_gen2))
        logger.info(' ')
        logger.info('epoch : %d =====================================================' %(epoch))
        logger.info(' ')
        test_time_end=time.time()
        df_res_tmp['train_time'] = train_time_end-train_time_start
        df_res_tmp['test_time'] = test_time_end-test_time_start
        df_res_tmp['epoch'] = epoch
        df_res_tmp['train_acc'] = pd.DataFrame(acc0).mean().values
        df_res_tmp['test_acc'] = pd.DataFrame(acc_test).mean().values
        df_res_tmp['test_age_acc'] = acc_age2
        df_res_tmp['test_gen_acc'] = acc_gen2
        df_res_tmp['test_sum_acc'] = acc_age2+acc_gen2
        df_res.append(df_res_tmp)
    
    nname = models+'_s'+str(seeddd)+'_tr'+str(tr_layer)+'_lstm'+str(lstm_layer)+'_gru'+str(gru_layer)+'_rnn'+str(rnn_layer)+'_dnn'+str(dnn_dim)
    save_model_name = 'model/' + nname
    torch.save(model_lstm.state_dict(), path_tmpdata / save_model_name)
    
    df_res=pd.DataFrame(df_res)
    total_params = sum(p.numel() for p in model_lstm.parameters())
    total_trainable_params = sum(p.numel() for p in model_lstm.parameters() if p.requires_grad)
    df_res['model_name'] = save_model_name
    df_res['total parameters'] = total_params
    df_res['training parameters'] = total_trainable_params
    
    save_df_name = 'output/' + nname + '.csv'
    df_res.to_csv(path_tmpdata / save_df_name)
    
    # 清理cuda
    model_lstm.to('cpu')
    del model_lstm
    torch.cuda.empty_cache()
    time.sleep(5*60)
