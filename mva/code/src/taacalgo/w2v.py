# -*- coding: utf-8 -*-
# @author: 
# @email: nngyusong@gmail.com
# @date: 2020/07/22

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from gensim.models import Word2Vec
from .util import path_tmpdata
from .util import logger

def get_index(sentence, word_index):
    sequence = []
    for word in sentence:
        try:
            sequence.append(word_index[word])
        except KeyError:
            pass
    return sequence

def train_w2v(behavior_list, adtype, win, negative=5, workers=20, epoch_=8):

    """ 训练w2v 保存embedding_mat

    :param behavior_list: 行为序列
    :param adtype: 粒度名
    :param win: 窗口大小
    :param negative: 负采样数
    :return:

    """

    # log
    logger.info('start train word2vec type:%s windows:%d' %(adtype, win))

    # train w2v
    model = Word2Vec(size=512, window=win, min_count=1, sg=1, workers=workers, negative=negative)
    model.build_vocab(behavior_list)  # prepare the model vocabulary
    model.train(behavior_list, total_examples=model.corpus_count, epochs = epoch_)

    # save embedding_mat
    embedding_matrix = torch.from_numpy(model.wv.vectors)
    save_name = 'embed_' + adtype + '_w' + str(win)
    save_path_embed_mat = path_tmpdata / save_name
    torch.save(embedding_matrix, save_path_embed_mat)


def train_w2v_get_pad(behavior_list,adtype,win,negative=5,workers=20, epoch_=8):

    """ train w2v & save embedding_mat pad_sequence

    :param behavior_list: 行为序列
    :param adtype: 粒度名
    :param win: 窗口大小
    :param negative: 负采样数
    :return:

    """

    # log
    logger.info('start get pad_sequence train word2vec type:%s windows:%d' %(adtype, win))

    # train w2v
    model = Word2Vec(size=512, window=win, min_count=1, sg=1, workers=workers, negative=negative)
    model.build_vocab(behavior_list)  # prepare the model vocabulary
    model.train(behavior_list, total_examples=model.corpus_count, epochs=epoch_)

    # save embedding_mat
    embedding_matrix = model.wv.vectors
    embedding_matrix = torch.from_numpy(embedding_matrix)
    save_name = 'embed_' + adtype + '_w' + str(win)
    save_path_embed_mat = path_tmpdata / save_name
    torch.save(embedding_matrix, save_path_embed_mat)

    # save pad_sequence -128
    vocab_list = list(model.wv.vocab.items())
    word_index = {word[0]: word[1].index for word in vocab_list}
    X_data = list(map(lambda x: get_index(x,word_index), behavior_list))
    X_data_pack = [torch.from_numpy(np.array(x[-128:])) for x in X_data]
    X_len = [len(x) for x in X_data_pack]
    X_len = torch.from_numpy(np.array(X_len))
    X_pad = pad_sequence(X_data_pack, batch_first=True)
    save_name_len = 'pad_len_-128'
    save_path_len = path_tmpdata / save_name_len
    save_name_pad = 'pad_'+adtype+'_-128'
    save_path_pad = path_tmpdata / save_name_pad
    torch.save(X_pad, save_path_pad)
    torch.save(X_len, save_path_len)

    # save pad_sequence +128
    X_data_pack = [torch.from_numpy(np.array(x[:128])) for x in X_data]
    X_len = [len(x) for x in X_data_pack]
    X_len = torch.from_numpy(np.array(X_len))
    X_pad = pad_sequence(X_data_pack, batch_first=True)
    save_name_len = 'pad_len_+128'
    save_path_len = path_tmpdata / save_name_len
    save_name_pad = 'pad_'+adtype+'_+128'
    save_path_pad = path_tmpdata / save_name_pad
    torch.save(X_pad, save_path_pad)
    torch.save(X_len, save_path_len)

def get_pad_clicktime(behavior_time):

    """ save pad_clicktime

    :param behavior_list: 行为序列
    :return:

    """

    # save pad_clicktime -128
    behavior_time_pack = [torch.from_numpy(np.array(x[-128:])) for x in behavior_time]
    behavior_time_pad = pad_sequence(behavior_time_pack, batch_first=True)
    save_path_clicktime = path_tmpdata / 'pad_clicktime_-128'
    torch.save(behavior_time_pad, save_path_clicktime)

    # save pad_clicktime +128
    behavior_time_pack = [torch.from_numpy(np.array(x[:128])) for x in behavior_time]
    behavior_time_pad = pad_sequence(behavior_time_pack, batch_first=True)
    save_path_clicktime = path_tmpdata / 'pad_clicktime_+128'
    torch.save(behavior_time_pad, save_path_clicktime)
