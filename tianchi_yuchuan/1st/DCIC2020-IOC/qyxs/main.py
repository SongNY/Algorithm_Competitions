#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:juzphy
# datetime:2020/2/29 6:05 下午
from config import config
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['PYTHONHASHSEED'] = '0'


def hashfxn(astring):
    return ord(astring[0])


def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)


def get_geohash_tfidf(df, group_id, group_target, num):
    df[group_target] = df.apply(
        lambda x: geohash_encode(x['lat'], x['lon'], 7),
        axis=1)
    tmp = df.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_enc_tmp = TfidfVectorizer()
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(tmp[group_target])
    svd_tag_tmp = TruncatedSVD(n_components=num, n_iter=20, random_state=1024)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = ['{}_tfidf_{}'.format(group_target, i)
                           for i in range(num)]

    countvec = CountVectorizer()
    count_vec_tmp = countvec.fit_transform(tmp[group_target])
    svd_tmp = TruncatedSVD(n_components=num, n_iter=20, random_state=1024)
    svd_tmp = svd_tmp.fit_transform(count_vec_tmp)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(group_target, i)
                       for i in range(num)]

    return pd.concat([tmp[[group_id]], tag_svd_tmp, svd_tmp], axis=1)


def get_grad_tfidf(df, group_id, group_target, num):
    grad_df = df.groupby(group_id)['lat'].apply(
        lambda x: np.gradient(x)).reset_index()
    grad_df['lon'] = df.groupby(group_id)['lon'].apply(
        lambda x: np.gradient(x)).reset_index()['lon']
    feat_list = [group_id]
    grad_df['lat'] = grad_df['lat'].apply(lambda x: np.round(x, 4))
    grad_df['lon'] = grad_df['lon'].apply(lambda x: np.round(x, 4))
    grad_df[group_target] = grad_df.apply(lambda x: ' '.join([
        '{}_{}'.format(z[0], z[1]) for z in zip(x['lat'], x['lon'])
    ]),
                                          axis=1)

    tfidf_enc_tmp = TfidfVectorizer()
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(grad_df[group_target])
    svd_tag_tmp = TruncatedSVD(n_components=num, n_iter=20, random_state=1024)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = ['{}_tfidf_{}'.format(group_target, i)
                           for i in range(num)]
    return pd.concat([grad_df[feat_list], tag_svd_tmp], axis=1)


def get_sample_tfidf(df, group_id, group_target, num):
    tmp = df.groupby(group_id)['lat_lon'].apply(
        lambda x: x.sample(frac=0.1,
                           random_state=1)).reset_index()
    del tmp['level_1']
    tmp.columns = [group_id, group_target]
    tmp = tmp.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_enc_tmp = TfidfVectorizer()
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(tmp[group_target])
    svd_tag_tmp = TruncatedSVD(n_components=num, n_iter=20, random_state=1024)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = ['{}_tfidf_{}'.format(group_target, i)
                           for i in range(num)]

    return pd.concat([tmp[[group_id]], tag_svd_tmp], axis=1)


def w2v_feat(df, group_id, feat, length):
    print('start word2vec ...')
    data_frame = df.groupby(group_id)[feat].agg(list).reset_index()
    model = Word2Vec(data_frame[feat].values,
                     size=length,
                     window=5,
                     min_count=1,
                     workers=1,
                     iter=10,
                     seed=1,
                     hashfxn=hashfxn)
    data_frame[feat] = data_frame[feat].apply(lambda x: pd.DataFrame([
        model[c] for c in x
    ]))
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(
            lambda x: x[m].mean())
    del data_frame[feat]
    return data_frame


def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q40(x):
    return x.quantile(0.4)


def q60(x):
    return x.quantile(0.6)


def q70(x):
    return x.quantile(0.7)


def q80(x):
    return x.quantile(0.8)


def q90(x):
    return x.quantile(0.9)


def gen_feat(df):
    df.sort_values(['ID', 'time'], inplace=True)

    df['time'] = df['time'].apply(lambda x: '2019-' + x.split(
        ' ')[0][:2] + '-' + x.split(' ')[0][2:] + ' ' + x.split(' ')[1])
    df['time'] = pd.to_datetime(df['time'])

    df['lat_diff'] = df.groupby('ID')['lat'].diff(1)
    df['lon_diff'] = df.groupby('ID')['lon'].diff(1)
    df['speed_diff'] = df.groupby('ID')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('ID')['time'].diff(1).dt.seconds // 60
    df['anchore'] = df.apply(lambda x: 1
                             if x['lat_diff'] < 0.01 and x['lon_diff'] < 0.01
                             and x['speed'] < 0.1 and x['diff_minutes'] <= 10
                             else 0,
                             axis=1)

    gt_zero = df[(df['lat_diff'] != 0) & (df['lat_diff'] != 0)]
    speed_gt_zero = df[df['speed_diff'] != 0]

    df['type'] = df['type'].map({'围网': 0, '刺网': 1, '拖网': 2, 'unknown': -1})
    group_df = df.groupby('ID')['type'].agg(
        {'label': 'mean',
         'cnt': 'count'}).reset_index()

    anchore_df = df.groupby('ID')['anchore'].agg('sum').reset_index()
    anchore_df.columns = ['ID', 'anchore_cnt']

    group_df = group_df.merge(anchore_df, on='ID', how='left')
    group_df['anchore_ratio'] = group_df['anchore_cnt'] / group_df['cnt']

    stat_functions = ['min', 'max', 'mean', 'median', 'nunique', q10, q20, q30,
                      q40, q60, q70, q80, q90]
    stat_ways = ['min', 'max', 'mean', 'median', 'nunique', 'q_10', 'q_20',
                 'q_30', 'q_40', 'q_60', 'q_70', 'q_80', 'q_90']

    stat_cols = ['lat', 'lon', 'speed', 'direction']
    group_tmp = df.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
    group_tmp.columns = ['ID'] + ['{}_{}'.format(i, j)
                                  for i in stat_cols for j in stat_ways]

    gt_zero_group = gt_zero.groupby(
        'ID')[stat_cols].agg(stat_functions).reset_index()
    gt_zero_group.columns = ['ID'] + ['pos_gt_zero_{}_{}'.format(i, j)
                                      for i in stat_cols for j in stat_ways]

    speed_gt_zero_group = speed_gt_zero.groupby(
        'ID')[stat_cols].agg(stat_functions).reset_index()
    speed_gt_zero_group.columns = ['ID'] + ['speed_gt_zero_{}_{}'.format(
        i, j) for i in stat_cols for j in stat_ways]

    group_df = group_df.merge(group_tmp, on='ID', how='left')
    group_df = group_df.merge(gt_zero_group, on='ID', how='left')
    group_df = group_df.merge(speed_gt_zero_group, on='ID', how='left')

    mode_df = df.groupby(['ID', 'lat', 'lon'])['time'].agg(
        {'mode_cnt': 'count'}).reset_index()
    mode_df['rank'] = mode_df.groupby('ID')['mode_cnt'].rank(method='first',
                                                             ascending=False)
    for i in range(1, 4):
        tmp_df = mode_df[mode_df['rank'] == i]
        del tmp_df['rank']
        tmp_df.columns = ['ID', 'rank{}_mode_lat'.format(i),
                          'rank{}_mode_lon'.format(i),
                          'rank{}_mode_cnt'.format(i)]
        group_df = group_df.merge(tmp_df, on='ID', how='left')

    tfidf_df = get_geohash_tfidf(df, 'ID', 'lat_lon', 30)
    group_df = group_df.merge(tfidf_df, on='ID', how='left')
    print('geohash tfidf finished.')

    grad_tfidf = get_grad_tfidf(df, 'ID', 'grad', 30)
    group_df = group_df.merge(grad_tfidf, on='ID', how='left')
    print('gradient tfidf finished.')

    sample_tfidf = get_sample_tfidf(df, 'ID', 'sample', 30)
    group_df = group_df.merge(sample_tfidf, on='ID', how='left')
    print('sample tfidf finished.')

    w2v_df = w2v_feat(df, 'ID', 'lat_lon', 30)
    group_df = group_df.merge(w2v_df, on='ID', how='left')
    print('word2vec finished.')

    return group_df


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', scores, True


def sub_on_line_lgb(train_, test_, pred, label, cate_cols, split,
                    is_shuffle=True,
                    use_cart=False,
                    get_prob=False):
    n_class = 3
    train_pred = np.zeros((train_.shape[0], n_class))
    test_pred = np.zeros((test_.shape[0], n_class))
    n_splits = 5

    assert split in ['kf', 'skf'
                    ], '{} Not Support this type of split way'.format(split)

    if split == 'kf':
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=1024)
        kf_way = folds.split(train_[pred])
    else:
        folds = StratifiedKFold(n_splits=n_splits,
                                shuffle=is_shuffle,
                                random_state=1024)
        kf_way = folds.split(train_[pred], train_[label])

    print('Use {} features ...'.format(len(pred)))

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'num_class': n_class,
        'nthread': 8,
        'verbose': -1,
    }
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_[pred].iloc[train_idx
                                            ], train_[label].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx
                                            ], train_[label].iloc[valid_idx]

        if use_cart:
            dtrain = lgb.Dataset(train_x,
                                 label=train_y,
                                 categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x,
                                 label=valid_y,
                                 categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(params=params,
                        train_set=dtrain,
                        num_boost_round=3000,
                        valid_sets=[dvalid],
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        feval=f1_score_eval)
        train_pred[valid_idx] = clf.predict(valid_x,
                                            num_iteration=clf.best_iteration)
        test_pred += clf.predict(test_[pred],
                                 num_iteration=clf.best_iteration) / folds.n_splits
    print(classification_report(train_[label], np.argmax(train_pred,
                                                         axis=1),
                                digits=4))
    if get_prob:
        sub_probs = ['qyxs_prob_{}'.format(q) for q in ['围网', '刺网', '拖网']]
        prob_df = pd.DataFrame(test_pred, columns=sub_probs)
        prob_df['ID'] = test_['ID'].values
        return prob_df
    else:
        test_['label'] = np.argmax(test_pred, axis=1)
        return test_[['ID', 'label']]


def get_data(file_path, model):
    assert model in ['train',
                     'test'], '{} Not Support this type of file'.format(model)
    paths = os.listdir(file_path)
    tmp = []
    for t in tqdm(range(len(paths))):
        p = paths[t]
        with open('{}/{}'.format(file_path, p), encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                tmp.append(line.strip().split(','))
    tmp_df = pd.DataFrame(tmp)
    if model == 'train':
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time',
                          'type']
    else:
        tmp_df['type'] = 'unknown'
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time',
                          'type']
    tmp_df['lat'] = tmp_df['lat'].astype(float)
    tmp_df['lon'] = tmp_df['lon'].astype(float)
    tmp_df['speed'] = tmp_df['speed'].astype(float)
    tmp_df['direction'] = tmp_df['direction'].astype(int)
    return tmp_df


if __name__ == "__main__":
    TRAIN_PATH = config.train_dir
    TEST_PATH = config.test_dir
    PROB_PATH = config.prob_qyxs
    use_prob = True
    train = get_data(TRAIN_PATH, 'train')
    test = get_data(TEST_PATH, 'test')
    train = train.append(test)
    all_df = gen_feat(train)
    del train, test
    use_train = all_df[all_df['label'] != -1]
    use_test = all_df[all_df['label'] == -1]
    use_feats = [c for c in use_train.columns if c not in ['ID', 'label']]
    sub = sub_on_line_lgb(use_train, use_test, use_feats, 'label', [], 'kf',
                          is_shuffle=True,
                          use_cart=False,
                          get_prob=use_prob)
    if use_prob:
        sub.to_csv(PROB_PATH, encoding='utf-8', index=False)
    else:
        sub['label'] = sub['label'].map({0: '围网', 1: '刺网', 2: '拖网'})
        sub.to_csv(config.save_path,
                   encoding='utf-8',
                   header=None,
                   index=False)
