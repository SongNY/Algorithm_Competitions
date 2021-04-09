
import datetime
import os
from configparser import ConfigParser

import pandas as pd
from qcloud_cos import CosConfig, CosS3Client


def read_cos_config(filename=None):
    """读取cos配置文件
    可以指定配置文件路径，默认会读取~/.taac_cos.conf
    配置文件中需要有bucket、region、secret_id和secert_key这些信息
    """
    if filename is None:
        filename = "~/.taac_cos.conf"
    filename = os.path.expanduser(filename)
    with open(filename, "r") as f:
        config_str = f.read().strip()
    if not config_str.startswith("[cos]"):
        config_str = "[cos]\n" + config_str

    parser = ConfigParser()
    parser.read_string(config_str)
    config = {
        "bucket": parser["cos"]["bucket"].strip("\"'"),
        "region": parser["cos"]["region"].strip("\"'"),
        "secret_id": parser["cos"]["secret_id"].strip("\"'"),
        "secret_key": parser["cos"]["secret_key"].strip("\"'")
    }

    if not config["region"].startswith("ap-"):
        config["region"] = "ap-" + config["region"]
    return config


def upload(df, config=None, filename=None, dirname=None):
    """上传结果文件
    可以传入csv文件名，也可以直接传入pd.DataFrame
    """
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(df, pd.DataFrame):
        df = df[["user_id", "predicted_age", "predicted_gender"]]
    if config is None or isinstance(config, str):
        config = read_cos_config(config)

    if filename is None:
        time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
        filename = f"submission_{time_str}.csv"
    if dirname is None:
        dirname = ""
    full_filename = os.path.join(dirname, filename)

    cos_config = CosConfig(
        Region=config["region"],
        Secret_key=config["secret_key"],
        Secret_id=config["secret_id"]
    )
    client = CosS3Client(cos_config)
    client.put_object(
        Bucket=config["bucket"],
        Body=df.to_csv(index=False).encode(),
        Key=full_filename,
        EnableMD5=False
    )

    host = config["bucket"] + ".cos." + config["region"] + ".myqcloud.com"
    url = "https://" + host
    url = os.path.join(url, full_filename)
    return url
