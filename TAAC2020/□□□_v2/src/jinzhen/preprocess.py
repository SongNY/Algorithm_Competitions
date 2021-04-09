
import pickle

import numpy as np
import pandas as pd

from taac_var import (
    logger, path_cache, path_testdata,
    path_traindata_preliminary, path_traindata_semifinal
)


def load_click_log(force=False):
    """载入并保存点击日志
    读取训练集和测试集的点击日志，合并，按时间排序，保存为hdf
    默认情况下如果保存的文件已经存在就不会再次生成，如果设置force=True就会强制重新生成
    """
    logger.info("开始载入点击日志")
    filepath = path_cache / "click_log_df.h5"
    if filepath.exists() and not force:
        logger.info("点击日志文件已存在，跳过")
        return
    train_click_log_df1 = pd.read_csv(path_traindata_preliminary / "click_log.csv", dtype=np.int32)
    train_click_log_df2 = pd.read_csv(path_traindata_semifinal / "click_log.csv", dtype=np.int32)
    test_click_log_df = pd.read_csv(path_testdata / "click_log.csv", dtype=np.int32)
    click_log_df = pd.concat((train_click_log_df1, train_click_log_df2, test_click_log_df), 0)
    click_log_df["click_times"] = click_log_df["click_times"].astype(np.int16)
    click_log_df["time"] = click_log_df["time"].astype(np.int16)
    click_log_df = click_log_df.sort_values(["user_id", "time"])
    click_log_df.to_hdf(filepath, "df")
    logger.info("点击日志载入完成")
    return click_log_df


def load_ad_map(force=False):
    """生成广告素材id到其各种属性的映射
    默认情况下如果保存的文件已经存在就不会再次生成，如果设置force=True就会强制重新生成
    """

    logger.info("开始载入广告属性映射")
    filepath = path_cache / "ad_map.pkl"
    if filepath.exists() and not force:
        logger.info("广告属性映射文件已存在，跳过")
        return

    # 将product_id中的\N视为NA
    train_ad_df1 = pd.read_csv(path_traindata_preliminary / "ad.csv", na_values="\\N")
    train_ad_df2 = pd.read_csv(path_traindata_semifinal / "ad.csv", na_values="\\N")
    test_ad_df = pd.read_csv(path_testdata / "ad.csv", na_values="\\N")

    # NA只有可能是product_id中的\N，将其填充为-99
    ad_df = pd.concat((train_ad_df1, train_ad_df2, test_ad_df), 0).drop_duplicates().fillna(-99).astype(np.int32)
    ad_df["product_category"] = ad_df["product_category"].astype(np.int16)
    ad_df["industry"] = ad_df["industry"].astype(np.int16)
    ad_map = {
        "adid": dict(zip(ad_df["creative_id"], ad_df["ad_id"])),
        "prod": dict(zip(ad_df["creative_id"], ad_df["product_id"])),
        "cate": dict(zip(ad_df["creative_id"], ad_df["product_category"])),
        "ader": dict(zip(ad_df["creative_id"], ad_df["advertiser_id"])),
        "ind": dict(zip(ad_df["creative_id"], ad_df["industry"]))
    }
    with open(filepath, "wb") as f:
        pickle.dump(ad_map, f)
    logger.info("载入广告属性映射完成")
    return ad_map


def load_user_dict(force=False):
    """保存好点击日志中的素材id序列、时间序列、点击次数序列
    默认情况下如果保存的文件已经存在就不会再次生成，如果设置force=True就会强制重新生成
    """

    logger.info("开始载入每个用户的点击序列")
    path_user_dict = path_cache / "user_dict"
    path_user_dict.mkdir(exist_ok=True, parents=True)

    filepath_item = path_user_dict / "item.pkl"
    filepath_time = path_user_dict / "time.pkl"
    filepath_click_times = path_user_dict / "click_times.pkl"

    if filepath_item.exists() and not force:
        logger.info("每个用户的点击序列文件已存在，跳过")
        return

    path_click_log = path_cache / "click_log_df.h5"
    if not path_click_log.exists():
        click_log_df = load_click_log()
    else:
        click_log_df = pd.read_hdf(path_click_log)

    last_user_id = -1
    user_item_dict = {}
    user_time_dict = {}
    user_click_times_dict = {}
    for row in click_log_df.itertuples():
        # 点击日志已经按用户id排序
        # 如果在迭代过程中用户id发生变化
        # 就表示上一个用户的id已经结束
        if row.user_id != last_user_id:
            agg_list = [[], [], []]
            user_item_dict[row.user_id] = agg_list[0]
            user_time_dict[row.user_id] = agg_list[1]
            user_click_times_dict[row.user_id] = agg_list[2]
            last_user_id = row.user_id
        agg_list[0].append(row.creative_id)
        agg_list[1].append(row.time)
        agg_list[2].append(row.click_times)
    with open(filepath_item, "wb") as f:
        pickle.dump(user_item_dict, f)
    with open(filepath_time, "wb") as f:
        pickle.dump(user_time_dict, f)
    with open(filepath_click_times, "wb") as f:
        pickle.dump(user_click_times_dict, f)
    logger.info("载入每个用户的点击序列完毕")
    return user_item_dict, user_time_dict, user_click_times_dict


if __name__ == "__main__":
    load_ad_map()
    load_user_dict()
