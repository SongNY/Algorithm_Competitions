import decimal
import json
import os
import re
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path
import itertools
from bert_serving.client import BertClient
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Masking
from keras_contrib.layers import CRF

import numpy as np
import pandas as pd
import pdfplumber
import warnings
warnings.filterwarnings('ignore')

def debug(case, **kwargs):
    import pprint
    pprint.pprint(">>>> Debug Start")
    for name, value in kwargs.items():
        pprint.pprint(f">> {name} = ")
        pprint.pprint(value)
    pprint.pprint(f">> This is Case {case}")
    pprint.pprint(">>>> Debug End")
    input()

def training_dict_process(training_resume_dict):
    true_resume_dict = deepcopy(training_resume_dict)
    
    section_process_list = list({'教育经历', '项目经历', '工作经历'} & true_resume_dict.keys())
    for section_str in section_process_list:
        section_field_list = training_resume_dict[section_str][0].keys()
        for field_name in section_field_list:
            true_resume_dict.update({field_name: list(map(lambda item: item[field_name], training_resume_dict[section_str]))})
        del true_resume_dict[section_str]
    return true_resume_dict


def get_separator_list(words: list, width: int = 300) -> list:
    """
    将总pdf纵向划分为长度每为10的n个区域 
    判定n各区域有没有字
    无则将坐标放入separator_list
    """

    width = int(width)
    separator_list = []
    for separator in range(width // 10):
        separator = (separator + 1) * 10
        x1 = [str(word) for word in words if word["x1"] > separator]
        x2 = [str(word) for word in words if word["x0"] < separator]
        if x1 and x2:
            if not set(x1) & set(x2):
                separator_list.append(separator)
    return separator_list


def extract_words(page: pdfplumber.page.Page, i: int = 0) -> list:
    """
    自定义pdfplumber的extract_words函数
    """
    
    doctop_clusters = pdfplumber.utils.cluster_objects(page.chars, "doctop", 3)
    lines = []
    for line_chars in doctop_clusters:
        line_words = [[]]
        current_word = line_words[-1]
        current_word_font = ""
        line_chars = sorted(line_chars, key=lambda x: x["x0"])
        for char in line_chars:
            font_match = True
            char_size = char["bottom"] - char["top"]
            if char["text"].isspace():
                continue
            char_font = char["fontname"]
            if len(char["text"]) != 1:
                continue
            if ord(char["text"]) > 128 and current_word_font and char["text"] not in "（）。，：；":
                font_match = current_word_font == char_font
            if not current_word:
                current_word.append(char)
            elif char["x0"] - current_word[-1]["x1"] < char_size / 5 * 4 and font_match:
                current_word.append(char)
            else:
                current_word = [char]
                line_words.append(current_word)
                if ord(char["text"]) > 128:
                    current_word_font = char_font
        line_paste_words = []
        for word in line_words:
            if not word:
                continue
            font_counter = Counter(x["fontname"] for x in word)
            fontname = sorted(font_counter.items(), key=lambda x: x[1])[-1][0]
            paste_word = {
                "text": "".join(x["text"] for x in word),
                "x0": word[0]["x0"],
                "x1": word[-1]["x1"],
                "y0": min([x["top"] for x in word]) + 1000 * i,
                "y1": max([x["bottom"] for x in word]) + 1000 * i,
                "font": fontname
            }
            paste_word["size"] = paste_word["y1"] - paste_word["y0"]
            line_paste_words.append(paste_word)
        lines.append(line_paste_words)
    words = []
    for line_words in lines:
        words += line_words
    return words

def crf_split(line_list: list) -> dict:
    resume_info = {
        "job": [],
        "project": [],
        "edu": [],
        "base": [],
        'o':[]
    }
    word_list = []
    for line_id, line in enumerate(line_list):
        line = sorted(line, key=lambda x: (x['y0'], x['x0']))
        word_list += [[word, line_id] for word in line if word['text'] != '']
    word_text_list = [word[0]['text'] for word in word_list]
    word_encoded = bc.encode(word_text_list)

    resume_num = 1
    seq_maxlen = 190
    embedding_dim = 768
    word_paded = np.zeros((resume_num, seq_maxlen, embedding_dim)) 
    seq_len = len(word_text_list)
    if seq_len <= seq_maxlen:  # right padding
        word_paded[0, :seq_len, :] = word_encoded
        result_raw = model.predict(word_paded)[0][:seq_len,:] # len = seq_len
    else: 
        word_list = word_list[-seq_maxlen:]   # 切掉左边多余的，现在 len = seq_maxlen，从而保证 len(word_list) == len(result_tags)
        word_paded[0, :, :] = word_encoded[-seq_maxlen:]
        result_raw = model.predict(word_paded)[0] # len = seq_maxlen

    result = [np.argmax(row) for row in result_raw] 
    chunk_tags = ['base', 'edu', 'job', 'project','o']
    result_tags = [chunk_tags[i] for i in result] 

    word_list_tagged = [word_list[i] + [result_tags[i]] for i in range(len(word_list))]
    for section in ['base', 'job', 'edu', 'project','o']:
        word_list_section = [word[:2] for word in word_list_tagged if word[-1]==section]
        line_list_section = [[item[0] for item in list(b)] for _, b in itertools.groupby(word_list_section, key=lambda x: x[1])]
        resume_info[section] = line_list_section
    return resume_info

def split_main_resume(line_list:list,break_mode,filename) -> dict:
    """
    将简历的行分为多个部分：基本信息、工作经历、项目经历、教育经历
    """

    resume_info = {
        "job": [],
        "project": [],
        "edu": [],
        "base": []
    }
    section_title_list = []

    useful_titles = {
        "job": ["主要经历", "工作经历", "工作经验", "工作背景",'工作信息','工作情况', "工作",'实习','实习经历','实习经验','实习信息','实习情况','专业实践','校外实习','校外实践','实习实践','实践实习','公司经历'],
        "project": ["项目经验", "项目经历", "项目背景",'项目信息', '项目情况',"项目" , "项目实践" , "实践项目"],
        "edu": ["教育背景", "教育经历",'教育水平','教育信息','教育情况', "教育",'学历','学历水平',"学习经历", "学历背景",'学历信息','学历情况','学习简历']
    }
    all_useful_titles = sum(useful_titles.values(), [])

    ## 先判定每每行只有一个字符串 然后匹配 四个字 and 去字符后字数相等
    ## 例如 工作描述: 不会被匹配进来
    for line in line_list:
        if len(line) > 1:
            continue
        line_text = re.sub("[:：。；，.;0-9a-zA-Z司]", "", line[0]["text"])
        if len(line[0]["text"]) == 4 and len(line[0]["text"]) == len(line_text):
            section_title_list.append(line[0])

    ##如果没找到继续匹配两个字 (要不要匹配6个字?) or len(line[0]["text"]) == 6
#     if not section_title_list:
#         for line in line_list:
#             if len(line) > 1:
#                 continue
#             line_text = re.sub("[:：。；，.;0-9a-zA-Z司]", "", line[0]["text"])
#             if (len(line[0]["text"]) == 2) and len(line[0]["text"]) == len(line_text):
#                 section_title_list.append(line[0])

    if not section_title_list:
        print(f"{filename}4字单行section_title_list未匹配到,尝试文字匹配")
        for line in line_list:
            line_text = re.sub("[:：。；，.;0-9a-zA-Z\(\)（）\[\]]", "", line[0]["text"])
            if line_text not in all_useful_titles:
                continue
            else:
                section_title_list.append(line[0])

    if not section_title_list and break_mode:
        return
    
    if not section_title_list:
        print(f"{filename}关闭separator模式下提取的简历信息发生错误,启用CRF模型")
        return crf_split(line_list)

#     meet_type=set([x['text'] for x in section_title_list]).intersection(set(all_useful_titles))
#     if not meet_type:
#         print(f"{filename}关闭separator,meet no section title,启用CRF模型")
#         return crf_split(line_list)
    
    if section_title_list:
        ## 限制字体
        font_counter = Counter([x["font"] for x in section_title_list])
        real_section_title_font = sorted(font_counter.items(), key=lambda x: x[1])[-1][0]
        section_title_list = [x for x in section_title_list if x["font"] == real_section_title_font]

        ## 限制字号
        size_counter = Counter([x["size"] for x in section_title_list])
        real_section_title_size = sorted(size_counter.items(), key=lambda x: x[1])[-1][0]
        section_title_list = [x for x in section_title_list if abs(x["size"] - real_section_title_size) < 0.3]

    ## 排序去重
    section_title_list_new = []
    for word in sorted(section_title_list, key=lambda x: x["y0"]):
        if section_title_list_new and word["text"] in [x["text"] for x in section_title_list_new]:
            continue
        section_title_list_new.append(word)
    section_title_list = section_title_list_new

    ## 提取切分点 加’基本信息‘ 切分标签
    section_split_point = [x["y0"] - decimal.Decimal("0.1") for x in section_title_list]
    cut_bins = [0] + section_split_point + [99999]
    cut_labels = ["基本信息"] + [x["text"] for x in section_title_list]

    ## 去’基本信息‘重复
    if (len(cut_labels)>1):
        if (cut_labels[0]==cut_labels[1]):
            del cut_labels[1]
            del cut_bins[1]
    
    ## 切割
    line_y0_series = pd.Series([min(x["y0"] for x in y) for y in line_list]).astype(float)
    line_type_list = pd.cut(line_y0_series, bins=cut_bins, labels=cut_labels).tolist()

    meet_useful_title = False
    for i, line in enumerate(line_list):
        line_type = line_type_list[i]
        if line_type in all_useful_titles:
            meet_useful_title = True
        if line_type in useful_titles["job"]:
            resume_info["job"].append(line)
        elif line_type in useful_titles["project"]:
            resume_info["project"].append(line)
        elif line_type in useful_titles["edu"]:
            resume_info["edu"].append(line)
        elif not meet_useful_title:
            resume_info["base"].append(line)

    return resume_info


def get_main_line_list(word_list: list) -> list:
    """
    将同一行的字符串放到一起，并处理部分因换行而被拆分开的字符串
    """
    word_list = deepcopy(word_list)
    line_list = []
    current_line_y1 = -1
    current_line = []
 
    # 初步将同一行的字符串放到一起
    for word in word_list:
        if word["y0"] > current_line_y1:
            line_list.append([])
            current_line = line_list[-1]
        else:
            current_line.append(word)
            current_line_y1 = max(x["y1"] for x in current_line)
        if not current_line:
            current_line.append(word)
            current_line_y1 = word["y1"]

    # 尝试拼接因换行而被拆分开的字符串
    ## 左/右对齐+字体(除去第四位)字号相同+下一行字数小于上一行
    ## eg22f5b9437b47 右对齐:营销储备干部7500/起 左对齐:ATCHAIN上海艺巢信息技术有/限公司
    new_line_list = []
    def dic_key(dic):
        return dic['x0']
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if not new_line_list or len(new_line_list[-1]) < 2:
            new_line_list.append(line)
            continue
        if line[0]["x0"] - min(new_line_list[-1],key=dic_key)['x0'] < line[0]["size"] * 5:
            new_line_list.append(line)
            continue
        append_word_list=[]
        for word in line:
            for last_line_word in new_line_list[-1]:
                is_left_align = abs(word["x0"] - last_line_word["x0"]) < 0.1
                is_right_align = abs(word["x1"] - last_line_word["x1"]) < 0.2
                is_font_match = word['font'][0:3]+word['font'][4:] == last_line_word["font"][0:3]+last_line_word['font'][4:]
                is_size_match = abs(word["size"] - last_line_word["size"]) < 0.1
                is_num_large = len(word["text"]) - len(last_line_word["text"]) < 0
                if is_left_align and is_font_match and is_size_match and is_num_large:
                    last_line_word["text"] += word["text"]
                    break
                if is_right_align and is_font_match and is_size_match and is_num_large:
                    last_line_word["text"] += word["text"]
                    break
            if not ((is_left_align or is_right_align) and is_font_match and is_size_match and is_num_large):
                append_word_list.append(word)
        if append_word_list:
            new_line_list.append(append_word_list)
    line_list = new_line_list
    
    ## 一种特殊情况
    ## 如02b5bd8a71b6的工作经历的时间被拆为多行
    ## 且公司名称的字符串的位置卡在两行正中间
    ## 解决至今字体大小不同问题
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        new_line = []
        for i, word in enumerate(line):
            if not i:
                new_line.append(word)
                continue
            is_left_align = abs(word["x0"] - line[i - 1]["x0"]) < 0.1
            is_font_match = word['font'][0:3]+word['font'][4:] == line[i - 1]["font"][0:3]+line[i - 1]['font'][4:]
            is_size_match = abs(word["size"] - line[i - 1]["size"]) < 0.1
            is_num_large = len(word["text"]) - len(line[i - 1]["text"]) < 0
            time_re = "(.*?)((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)(.*?)$"
            is_time_match = re.match(time_re, line[i - 1]["text"])
            is_jin_match = re.match("(.*?)(今|现在)$", word["text"])
            if is_left_align and is_font_match and is_size_match and is_num_large:
                line[i - 1]["text"] += word["text"]
            elif is_time_match and is_jin_match and is_font_match:
                line[i - 1]["text"] += word["text"]
            else:
                new_line.append(word)
        new_line_list.append(new_line)
    line_list = new_line_list

    ## 处理第二种特殊情况 eg0fa73714b05b
    ## 行内2元素 时间公司连到一起
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if len(line) != 2:
            new_line_list.append(line)
            continue
        time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
        time_range_match = re.match(time_re + "(.*?)" + f"({time_re}|今)", line[0]["text"])
        if time_range_match is None:
            new_line_list.append(line)
            continue
        company_re = f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)$"
        company_match = re.findall(company_re,re.sub(time_range_match.group(0),'',line[0]['text']))
        if not company_match:
            new_line_list.append(line)
            continue
        new_line =  deepcopy([line[0]])+deepcopy(line)
        new_line[0]['text'] = time_range_match.group(0)
        new_line[1]['text'] = re.sub(time_range_match.group(0),'',new_line[1]['text'])
        new_line_list.append(new_line)  
    line_list = new_line_list

    ## 处理453a872caa0e
    ## 行内3元素 时间公司职位连到一起
    ## 处理BOSS直聘
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if len(line) != 1:
            new_line_list.append(line)
            continue
        if line[0]['text'] == '简历来自：BOSS直聘':
	        continue
        time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
        time_range_match = re.match(time_re + "(.*?)" + f"({time_re}|今)", line[0]["text"])
        if time_range_match is None:
            new_line_list.append(line)
            continue
        company_re = f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)"
        no_time_line=re.sub(time_range_match.group(0),'',line[0]['text'])
        company_match = re.findall(company_re,no_time_line)
        if not company_match:
            new_line_list.append(line)
            continue
        no_time_noone_company_line=re.sub(company_match[0][0],'',no_time_line)
        double_company_match = re.findall(company_re,no_time_noone_company_line)
        new_line = deepcopy(line)+ deepcopy(line)+deepcopy(line)
        new_line[0]['text'] = time_range_match.group(0)
        if double_company_match:
            new_line[1]['text'] = company_match[0][0]+double_company_match[0][0]
            new_line[2]['text'] = re.sub(double_company_match[0][0],'',no_time_noone_company_line)
        else:
            new_line[1]['text'] = company_match[0][0]
            new_line[2]['text'] = no_time_noone_company_line
        new_line_list.append(new_line)  
    line_list = new_line_list
    
    ## 处理0fa73714b05b 3+1且职位跨行
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if not new_line_list or len(new_line_list[-1]) != 3:
            new_line_list.append(line)
            continue
        if len(line)!=1 or line[0]['size']!=decimal.Decimal('11.757'):
            new_line_list.append(line)
            continue
        if len(line[0]["text"]) - len(new_line_list[-1][-1]["text"]) >= 0:
            new_line_list.append(line)
            continue
        word=line[0]
        time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
        is_left_time=re.match(time_re + "(.*?)" + f"({time_re}|今)$", new_line_list[-1][0]['text'])
        company_re = f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)$"
        is_company_match = re.match(company_re,new_line_list[-1][1]['text'])
        is_left_align = abs(word["x0"] - new_line_list[-1][0]["x0"]) < 0.1
        is_font_match = word['font'][0:3]+word['font'][4:] == new_line_list[-1][-1]["font"][0:3]+new_line_list[-1][-1]['font'][4:]
        is_size_match = abs(word["size"] - new_line_list[-1][-1]["size"]) < 0.1
        if is_left_align and is_font_match and is_size_match and is_left_time and is_company_match:
            new_line_list[-1][-1]["text"] += word["text"]
            continue
        else:
            new_line_list.append(line)
    line_list = new_line_list

    ## 处理a574380a889e/31caeb4dd171 2+1 项目名称跨行
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if not new_line_list or len(new_line_list[-1]) != 2:
            new_line_list.append(line)
            continue
        if len(line)!=1 or (line[0]['size']!=decimal.Decimal('9.585') and line[0]['size']!=decimal.Decimal('10.560')):
            new_line_list.append(line)
            continue
        if len(line[0]["text"]) >= 3:
            new_line_list.append(line)
            continue
        word=line[0]
        time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
        is_left_time=re.match(time_re + "(.*?)" + f"({time_re}|今)$", new_line_list[-1][0]['text'])
        is_right_time=re.match(time_re + "(.*?)" + f"({time_re}|今)$", new_line_list[-1][1]['text'])
        is_left_align = abs(word["x0"] - new_line_list[-1][0]["x0"]) < 0.1
        is_font_match = word['font'][0:3]+word['font'][4:] == new_line_list[-1][-1]["font"][0:3]+new_line_list[-1][-1]['font'][4:]
        is_size_match = abs(word["size"] - new_line_list[-1][-1]["size"]) < 0.1
        if is_left_align and is_font_match and is_size_match and is_left_time:
            new_line_list[-1][-1]["text"] += word["text"]
            continue
        if is_left_align and is_font_match and is_size_match and is_right_time:
            new_line_list[-1][0]["text"] += word["text"]
            continue
        else:
            new_line_list.append(line)
    line_list = new_line_list

    ## 处理c8aaf14994ed 硕|博|学士学 位字跨行
    new_line_list = []
    for line in line_list:
        line = deepcopy(sorted(line, key=lambda x: x["x0"]))
        if len(line) == 3:
            if line[2]['text']=='硕士学' or line[2]['text']=='博士学' or line[2]['text']=='学士学':
                new_line=deepcopy(line)
                new_line[2]['text']+='位'
                new_line_list.append(new_line)
                continue
        new_line_list.append(line)
    line_list = new_line_list

    return line_list

def gather_base_info(resume_info: dict) -> str:
    new_base_info = []
    
    #  整理side模块内word 
    ## 拼接因换行而被拆分开的字符串
    ## 1右对齐+字体字号匹配 eg755b09bd4ef7 
    ## 2左对齐+字体字号匹配+上个词有冒号+新词无冒号 除去了eg9ee54deb176c
    for word in resume_info["side"]:
        word = deepcopy(word)
        if not new_base_info:
            new_base_info.append([word])
            continue
        lastword = new_base_info[-1][0]
        is_right_align = abs(word["x1"] - lastword["x1"]) < 0.1
        is_left_align = abs(word["x0"] - lastword["x0"]) < 0.1
        is_font_match = word["font"] == lastword["font"]
        is_size_match = abs(word["size"] - lastword["size"]) < 0.1
        if is_right_align and is_font_match and is_size_match:
            new_base_info[-1][0]["text"] += word["text"]
        elif is_left_align and is_font_match and is_size_match and \
                lastword["text"].find("：") > -1 and word["text"].find("：") == -1:
            new_base_info[-1][0]["text"] += word["text"]
        else:
            new_base_info.append([word])
    
    #  整理base模块内word 
    ## 解决d0f8361c482d换行 未解决54ad6d16f9c2邮箱空格过大问题
    ## 合并条件:一个字符串 字符数<=3 字号相等
    for line in resume_info["base"]:
        line = deepcopy(line)
        if len(line) > 1 or len(line[0]["text"]) > 3 or not new_base_info:
            new_base_info.append(line)
        elif line[0]["size"] != new_base_info[-1][0]["size"]:
            new_base_info.append(line)
        else:
            for word in new_base_info[-1]:
                if word["x0"] == line[0]["x0"]:
                    word["text"] = word["text"] + line[0]["text"]
                    break
    #  转换为 a|b 去掉一些字符
    base_info_str = "|".join("|".join(x["text"] for x in y) for y in new_base_info)
    base_info_str = base_info_str.replace("：", "|").replace(":", "|")
    base_info_str = re.sub("\\|/(\\||$)", "", base_info_str)
    base_info_str = re.sub("\\|+", "|", base_info_str)
    return new_base_info, base_info_str


def extract_base_info(resume_info: dict, normal_word_size: decimal.Decimal) -> dict:
    res = {}
    new_base_info, base_info_str = gather_base_info(resume_info)

    #  初步尝试获取姓名
    ## 获取最大宽度
    base_info_line_size = []
    for line in new_base_info:
        line_y1 = max(x["y1"] for x in line)
        line_y0 = max(x["y0"] for x in line)
        base_info_line_size.append(line_y1 - line_y0)
    name_line = new_base_info[np.argmax(base_info_line_size)]

    #  依次降低规则寻找名字
    ## 1宽度最大列 一个字符串 字符2-4 且不包含简历而字 ##加入单字姓名
    ## 2一个字符串 字符2-4 且不包含简历而字
    ## 3第一个字符串比这一列字体大 字符2-4
    name0 = ""
    if len(name_line) == 1 and 2 <= len(name_line[0]["text"]) <= 4 and name_line[0]["text"].find("简历") == -1:
        name0 = name_line[0]["text"]
    elif len(name_line) == 1 and 1 == len(name_line[0]["text"]) and name_line[0]["text"].find("简历") == -1:
        name0 = name_line[0]["text"]
    else:
        for line in new_base_info:
            if len(line) == 1 and 2 <= len(line[0]["text"]) <= 4 and line[0]["text"].find("简历") == -1:
                name0 = line[0]["text"]
                break
            line_large_fontsize = [(x["size"] - decimal.Decimal("0.1")) > normal_word_size for x in line]
            if line_large_fontsize[0] and not any(line_large_fontsize[1:]) and 2 <= len(line[0]["text"]) <= 4:
                name0 = line[0]["text"]
                break

    # 开始提取各种基本信息
    name = re.findall("姓(\\||)名\\|([^\\|]+?)(\\||$)", base_info_str)
    if name:
        name = re.findall("'(.*?)'", str(name))
        name = sorted(name, key=lambda x: len(x))[-1]
        res["姓名"] = name
    else:
        # 如果没有全文没有“姓名”二字，则使用之前初步提取的姓名
        res["姓名"] = name0

    birthday = re.findall("((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)", base_info_str)
    if birthday:
        birthday = re.findall("'(.*?)'", str(birthday))
        birthday = sorted(birthday, key=lambda x: len(x))[-1]
        res["出生年月"] = birthday

    gender = re.findall("((\\||^)(男|女)(\\||$))", base_info_str)
    if gender:
        gender = re.findall("(男|女)", str(gender))[-1]
        res["性别"] = gender

    phone = re.findall("(\\||^)(1[0-9]{10})(\\||$)", base_info_str)
    if phone:
        phone = re.findall("'(.*?)'", str(phone))
        phone = sorted(phone, key=lambda x: len(x))[-1]
        res["电话"] = phone

    domicile = re.findall("(户(\\||)籍|户口/国籍|户(\\||)口)\\|([^\\|]+?)(\\||$)", base_info_str)
    if domicile:
        domicile = re.findall("'(.*?)'", str(domicile))
        domicile = [x for x in domicile if x not in ["户籍", "户口/国籍", "户口"]]
        domicile = sorted(domicile, key=lambda x: len(x))[-1]
        res["落户市县"] = domicile

    if base_info_str.find("户口/国籍") == -1:
        base_info_str = re.sub("(户(\\||)籍|户口/国籍|户(\\||)口)\\|([^\\|]+?)(\\||$)", "", base_info_str)
        address = re.findall("籍(\\||)贯\\|([^\\|]+?)(\\||$)", base_info_str)
        if not address:
            address = re.findall("(([^\\|]*?)市([^\\|]*?))", base_info_str)
        if address:
            address = re.findall("'(.*?)'", str(address))
            address = sorted(address, key=lambda x: len(x))[-1]
            res["籍贯"] = address

    degree = re.findall("(\\||^)(([^\\|]*?)(硕士|博士|本科|高中|初中|小学|大专|中专|研究生)(研究生|))(\\||$)", base_info_str)
    if degree:
        degree = re.findall("'(.*?)'", str(degree))
        degree = sorted(degree, key=lambda x: len(x))[-1]
        res["最高学历"] = degree
    else:
        edu_str = "|".join("|".join(x["text"] for x in y) for y in resume_info["edu"])
        degree = re.findall("学历/学位：([^/]+?)/", edu_str)
        if degree:
            res["最高学历"] = degree[0]

    political_face = re.findall("政治面貌\\|([^\\|]+?)(\\||$)", base_info_str)
    if political_face:
        political_face = re.findall("'(.*?)'", str(political_face))
        political_face = sorted(political_face, key=lambda x: len(x))[-1]
        res["政治面貌"] = political_face
    return res


def extract_edu_info(resume_info: dict) -> dict:
    time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
    degree_re = "((硕士|博士|本科|高中|初中|小学|大专|中专|研究生|学士)(研究生|)(学位|学历|))"
    combine_next_line = False
    edu_time_list = []
    college_list = []
    degree_list = []
    for i, line in enumerate(resume_info["edu"]):
        if combine_next_line:
            last_line = deepcopy(resume_info["edu"][i - 1])
            append_word_list=[]
            for word in line:
                for last_line_word in last_line:
                    is_left_align = abs(word["x0"] - last_line_word["x0"]) < 0.1
                    is_font_match = word["font"] == last_line_word["font"]
                    is_size_match = abs(word["size"] - last_line_word["size"]) < 0.1
                    is_num_large = len(word["text"]) - len(last_line_word["text"]) < 0
                    if is_left_align and is_font_match and is_size_match and is_num_large:
                        last_line_word["text"] += word["text"]
            for word in last_line:
                if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                    edu_time_list.append(word["text"])
                elif re.search("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?$", word["text"]):
                    college_list.append(word["text"])
                elif re.search(degree_re, word["text"]) is not None:
                    if re.search("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?(.*?)"+degree_re+'$',word["text"]) is not None:
                        college_list.append(re.sub(degree_re,"",word["text"]))
                        degree_list.append(re.sub("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?","",word["text"]))                
                    else:
                        degree_list.append(word["text"])
            combine_next_line = False
            continue

        line_str = "".join(x["text"] for x in line)
        single_time_match = re.match(time_re, line_str)
        time_range_match = re.match(time_re + "(.*?)" + f"({time_re}|今)", line_str)
        single_time_match_end = None
        for linel in line:
            if re.match('^'+time_re+'$',linel['text']) is not None:
                single_time_match_end = re.match('^'+time_re+'$',linel['text'])
        if single_time_match is not None and time_range_match is None and single_time_match_end is None:
            combine_next_line = True
            continue

        for word in line:
            if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                edu_time_list.append(re.split("[:：]", word["text"])[-1])
            elif re.search('^'+time_re+'$', word["text"]) is not None:
                edu_time_list.append(re.split("[:：]", word["text"])[-1])
            elif re.search("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?$", word["text"]) is not None:
                college_list.append(re.split("[:：]", word["text"])[-1])
            elif re.search(degree_re, word["text"]) is not None:
                if re.search("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?(.*?)"+degree_re+'$',word["text"]) is not None:
                    if word["text"] == "大学本科":
                        degree_list.append(word["text"])
                    else:
                        college_list.append(re.split("[\\|/:：]", re.sub(degree_re,"",word["text"]))[-1])
                        degree_list.append(re.split("[\\|/:：]", re.sub("(.*?)(大学|学院|校区|北大方正软件职业技术|天津市)(（.*?）)?(\\(.*?\\))?","",word["text"]))[-1])
                else:
                    degree_list.append(re.split("[\\|/:：]", word["text"])[-1])
    
    if not edu_time_list:
        for i, line in enumerate([resume_info["side"]]):
            edu_break_point=0
            for j,word in enumerate(line):
                edu_name = "教育背景|教育经历|教育"
                if re.search(edu_name,  word["text"]) is not None:
                    edu_break_point=j
                    break
            if edu_break_point!=0:
                for word in line[edu_break_point:]:                    
                    if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                        edu_time_list.append(re.split("[:：]", word["text"])[-1])
                    elif re.search('^'+time_re+'$', word["text"]) is not None:
                        edu_time_list.append(re.split("[:：]", word["text"])[-1])
                    elif re.search("(.*?)(大学|学院|校区|北大方正软件职业技术)(（.*?）)?(\\(.*?\\))?$", word["text"]) is not None:
                        college_list.append(re.split("[:：]", word["text"])[-1])
                    elif re.search(degree_re, word["text"]) is not None:
                        if re.search("(.*?)(大学|学院|校区|北大方正软件职业技术)(（.*?）)?(\\(.*?\\))?(.*?)"+degree_re+'$',word["text"]) is not None:
                            college_list.append(re.split("[\\|/:：]", re.sub(degree_re,"",word["text"]))[-1])
                            degree_list.append(re.split("[\\|/:：]", re.sub("(.*?)(大学|学院|校区|北大方正软件职业技术)(（.*?）)?(\\(.*?\\))?","",word["text"]))[-1])
                        else:
                            degree_list.append(re.split("[\\|/:：]", word["text"])[-1])
    
    for i,eduu_time in enumerate(edu_time_list):
        if len(re.findall(time_re, eduu_time))==2:
            edu_time_list[i]=re.findall(time_re, eduu_time)[1][0]
        elif len(re.findall(time_re, eduu_time))==1:
            edu_time_list.remove(eduu_time)
    return {
        "毕业时间": edu_time_list,
        "毕业院校": college_list,
        "学位": degree_list
    }


def extract_job_info(resume_info: dict) -> dict:
    job_time_list = []
    company_list = []
    jobname_list = []
    job_desc_list = []
    job_line_list = []
    job_word_tag_list = []
    combine_next_line = False
    time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
    for i, line in enumerate(resume_info["job"]):
        if combine_next_line:
            last_line = deepcopy(resume_info["job"][i - 1])
            job_line_list.append(last_line)
            job_word_tag_list.append([])
            append_word_list=[]
            for word in line:
                for last_line_word in last_line:
                    is_left_align = abs(word["x0"] - last_line_word["x0"]) < 0.1
                    is_font_match = word["font"] == last_line_word["font"]
                    is_size_match = abs(word["size"] - last_line_word["size"]) < 0.1
                    is_num_large = len(word["text"]) - len(last_line_word["text"]) < 0
                    if is_left_align and is_font_match and is_size_match and is_num_large:
                        last_line_word["text"] += word["text"]
                        break
                if not (is_left_align and is_font_match and is_size_match and is_num_large):
                    append_word_list.append(word)
            if append_word_list:
                job_line_list.append(append_word_list)
            for word in last_line:
                if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                    job_time = re.findall(f"({time_re}(.*?)({time_re}|今))", word["text"])[0][0]
                    job_time_list.append(job_time)
                    job_word_tag_list[-1].append("time")
                else:
                    job_word_tag_list[-1].append("")
            combine_next_line = False
            continue
        line_str = "".join(x["text"] for x in line)
        single_time_match = re.match(time_re, line_str)
        time_range_match = re.match(time_re + "(.*?)" + f"({time_re}|今)", line_str)
        single_time_match_end = None
        for linel in line:
            if re.match('^'+time_re+'$',linel['text']) is not None:
                single_time_match_end = re.match('^'+time_re+'$',linel['text'])
        if single_time_match is not None and time_range_match is None and single_time_match_end is None:
            combine_next_line = True
            continue

        job_line_list.append(line)
        job_word_tag_list.append([])
        for word in line:
            if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                job_time = re.findall(f"({time_re}(.*?)({time_re}|今))", word["text"])[0][0]
                job_time_list.append(re.split("[:：]", word["text"])[-1])
                job_word_tag_list[-1].append("time")
            elif re.search('^'+time_re+'$', word["text"]) is not None:
                job_time = re.findall(f"({time_re})", word["text"])[0][0]
                job_time_list.append(job_time)
                job_word_tag_list[-1].append("time")
            else:
                job_word_tag_list[-1].append("")

    time_line_id_list = [i for i, word_tag_line in enumerate(job_word_tag_list) if "time" in word_tag_line]
    company_line_id_list = []
    for time_id, time_line_id in enumerate(time_line_id_list):
        find_company = False
        word_list = [x["text"] for x in job_line_list[time_line_id]]
        for i, word in enumerate(word_list):
            re_res = re.findall(f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)$", word)
            if not re_res:
                re_res = re.findall("^((公司|单位)[:：](.*?))$", word)
            if re_res:
                company_line_id_list.append(time_line_id)
                job_word_tag_list[time_line_id][i] = "company"
                company_list.append(re.split("[:：]", re_res[0][0])[-1])
                find_company = True
        if find_company:
            continue
##job_error_1
        try:
            word_list = [x["text"] for x in job_line_list[time_line_id + 1]]
            for i, word in enumerate(word_list):
                re_res = re.findall(f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)$", word)
                if not re_res:
                    re_res = re.findall("^((公司|单位)[:：](.*?))$", word)
                if re_res:
                    company_line_id_list.append(time_line_id + 1)
                    job_word_tag_list[time_line_id + 1][i] = "company"
                    company_list.append(re.split("[:：]", re_res[0][0])[-1])
                    find_company = True
                else:
                    re_res = re.findall("^((公司|单位)[:：](.*?))$", word)
            if find_company and i:
                continue
        except BaseException:
            if (traceback.format_exc() not in traceback_list):
                traceback_list.append(traceback.format_exc())
                print(traceback.format_exc())

        word_list = [x["text"] for x in job_line_list[time_line_id - 1]]
        for i, word in enumerate(word_list):
            re_res = re.findall(f"^(([^\\.。，：；]*?)({company_last_two_pattern_str})(（.*?）)?(\\(.*?\\))?)$", word)
            if not re_res:
                re_res = re.findall("^((公司|单位)[:：](.*?))$", word)
            if re_res:
                company_line_id_list.append(time_line_id - 1)
                job_word_tag_list[time_line_id - 1][i] = "company"
                company_list.append(re.split("[:：]", re_res[0][0])[-1])
                find_company = True
        if find_company:
            continue
        if (len(job_line_list)>=time_line_id + 2):
            word_list = [
                re.search("^([^。，；.,]+?[^：。，；])$", word["text"]) for word in
                job_line_list[time_line_id - 1] + job_line_list[time_line_id] + job_line_list[time_line_id + 1]
                if word["text"] != job_time_list[time_id]
            ]
        else:
            word_list = [
                re.search("^([^。，；.,]+?[^：。，；])$", word["text"]) for word in
                job_line_list[time_line_id - 1] + job_line_list[time_line_id]
                if word["text"] != job_time_list[time_id]
            ]
        if not word_list:
            company_list.append("")
            company_line_id_list.append(time_line_id)
            continue
        word_len_list = [len(x.string) if x is not None else 0 for x in word_list]
        if not any(word_len_list):
            company_list.append("")
            company_line_id_list.append(time_line_id)
            continue
        company = word_list[np.argmax(word_len_list)].string
        company_list.append(re.split("[:：]", company)[-1])
        if (len(job_line_list)>=time_line_id + 2):
            for ii in [-1, 0, 1]:
                for jj, word in enumerate(job_line_list[time_line_id + ii]):
                    if word["text"].endswith(company):
                        company_line_id_list.append(time_line_id + ii)
                        job_word_tag_list[time_line_id + ii][jj] = "company"
        else:
            for ii in [-1, 0]:
                for jj, word in enumerate(job_line_list[time_line_id + ii]):
                    if word["text"].endswith(company):
                        company_line_id_list.append(time_line_id + ii)
                        job_word_tag_list[time_line_id + ii][jj] = "company"

    for time_id, time_line_id in enumerate(time_line_id_list):
        find_jobname = False
        if "company" in job_word_tag_list[time_line_id]:
            company_line_id = time_line_id
        elif (len(job_word_tag_list)>=time_line_id+2) and ("company" in job_word_tag_list[time_line_id + 1]):
            company_line_id = time_line_id + 1
        else:
            company_line_id = time_line_id - 1
        if len(job_line_list[time_line_id]) >= 3 and company_line_id == time_line_id:
##job_error_2
            try:
                i = job_word_tag_list[time_line_id].index("")
                job_word_tag_list[time_line_id][i] = "jobname"
                jobname = job_line_list[time_line_id][i]["text"]
                jobname_list.append(re.split("[:：]", jobname)[-1])
                continue
            except BaseException:
                print('job_error_2')
                if (traceback.format_exc() not in traceback_list):
                    traceback_list.append(traceback.format_exc())
                    print(traceback.format_exc())
        if not job_line_list[time_line_id] + job_line_list[company_line_id]:
            continue
        for word in job_line_list[time_line_id] + job_line_list[company_line_id]:
            if word["text"].endswith(job_time_list[time_id]) or word["text"].endswith(company_list[time_id]):
                continue
            re_res = re.findall("^职位[:：]([^。，；,]+?[^：。，])$", word["text"])
            if re_res:
                jobname_list.append(re_res[0])
                find_jobname = True
                break
        if find_jobname:
            continue        
        word_list = [
            word["text"] for word in job_line_list[time_line_id] + job_line_list[company_line_id]
            if not word["text"].endswith(job_time_list[time_id]) and not word["text"].endswith(company_list[time_id])
        ]
        if not word_list:
            continue
        word_len_list = [len(x) for x in word_list]
        jobname_list.append(word_list[np.argmax(word_len_list)])

    for i in range(len(time_line_id_list)):
        start_id = max([time_line_id_list[i], company_line_id_list[i]])
        if i + 1 < len(time_line_id_list):
            end_id = min([time_line_id_list[i + 1], company_line_id_list[i + 1]])
            job_desc_lines = job_line_list[(start_id + 1):end_id]
        else:
            job_desc_lines = job_line_list[(start_id + 1):]
        job_desc = "".join("".join(x["text"] for x in y) for y in job_desc_lines)
        job_desc = re.sub("^.{,2}(描述|内容|经历|职责)[:：]", "", job_desc)
        job_desc = re.sub("工作经历[:：]?$", "", job_desc)
        job_desc = re.sub("计算机软件\\|少于50人\\|民营公司工作描述：", "", job_desc)
        job_desc_list.append(job_desc)
    return {
        "工作时间": job_time_list,
        "工作单位": company_list,
        "工作内容": job_desc_list,
        "职务": jobname_list
    }


def extract_project_info(resume_info: dict) -> dict:
    project_time_list = []
    project_name_list = []
    project_line_list = []
    project_desc_list = []
    project_word_tag_list = []
    combine_next_line = False
    time_re = "((19|20)\\d\\d[年\\./](1[0-2]|0?[1-9])[月]?)"
    for i, line in enumerate(resume_info["project"]):
        if combine_next_line:
            last_line = deepcopy(resume_info["project"][i - 1])
            project_line_list.append(last_line)
            project_word_tag_list.append([])
            append_word_list=[]
            for word in line:
                for last_line_word in last_line:
                    is_left_align = abs(word["x0"] - last_line_word["x0"]) < 0.1
                    is_font_match = word["font"] == last_line_word["font"]
                    is_size_match = abs(word["size"] - last_line_word["size"]) < 0.1
                    is_num_large = len(word["text"]) - len(last_line_word["text"]) < 0
                    if is_left_align and is_font_match and is_size_match and is_num_large:
                        last_line_word["text"] += word["text"]
                        break
                if not (is_left_align and is_font_match and is_size_match and is_num_large):
                    append_word_list.append(word)
            if append_word_list:
                project_line_list.append(append_word_list)
            for word in last_line:
                if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                    project_time_list.append(re.split("[:：]", word["text"])[-1])
                    project_word_tag_list[-1].append("time")
                else:
                    project_word_tag_list[-1].append("")
            combine_next_line = False
            continue
        line_str = "".join(x["text"] for x in line)
        single_time_match = re.match(time_re, line_str)
        time_range_match = re.match(time_re + "(.*?)" + f"({time_re}|今)", line_str)
        single_time_match_end = None
        for linel in line:
            if re.match('^'+time_re+'$',linel['text']) is not None:
                single_time_match_end = re.match('^'+time_re+'$',linel['text'])
        if single_time_match is not None and time_range_match is None and single_time_match_end is None:
            combine_next_line = True
            continue

        project_line_list.append(line)
        project_word_tag_list.append([])
        for word in line:
            if re.search(time_re + "(.*?)" + f"({time_re}|今)", word["text"]) is not None:
                project_time_list.append(re.split("[:：]", word["text"])[-1])
                project_word_tag_list[-1].append("time")
            elif re.search('^'+time_re+'$', word["text"]) is not None:
                project_time_list.append(re.split("[:：]", word["text"])[-1])
                project_word_tag_list[-1].append("time")
            else:
                project_word_tag_list[-1].append("")

    time_line_id_list = [i for i, word_tag_line in enumerate(project_word_tag_list) if "time" in word_tag_line]
    name_line_id_list = []
    for time_id, time_line_id in enumerate(time_line_id_list):
        if len(project_line_list[time_line_id]) >= 2:
            word_list = [
                word["text"] for word in project_line_list[time_line_id]
                if not word["text"].endswith(project_time_list[time_id])
            ]
            word_len_list = [len(x) for x in word_list]
            project_name = word_list[np.argmax(word_len_list)]
            project_name = re.split("[:：]", project_name)[-1]
            if project_name == "UI设计师":
                project_name = project_line_list[time_line_id - 1][0]["text"]
                project_word_tag_list[time_line_id - 1][0] = "name"
                name_line_id_list.append(time_line_id - 1)
                project_name_list.append(project_name)
                continue
            project_name_list.append(project_name)

            for i, word in enumerate(project_line_list[time_line_id]):
                if word["text"].endswith(project_name_list[-1]):
                    project_word_tag_list[time_line_id][i] = "name"
                    name_line_id_list.append(time_line_id)
                    break
        elif project_line_list[time_line_id + 1][0]["text"].startswith("项目介绍"):
            project_name = project_line_list[time_line_id + 1][0]["text"]
            project_word_tag_list[time_line_id + 1][0] = "name"
            name_line_id_list.append(time_line_id + 1)
            project_name_list.append(re.split("[:：]", project_name)[-1])
        else:
            pass
##project_error
    if len(time_line_id_list)==len(name_line_id_list):
        for i in range(len(time_line_id_list)):
            start_id = max([time_line_id_list[i], name_line_id_list[i]])
            if i + 1 < len(time_line_id_list):
                end_id = min([time_line_id_list[i + 1], name_line_id_list[i + 1]])
                project_desc_lines = project_line_list[(start_id + 1):end_id]
            else:
                project_desc_lines = project_line_list[(start_id + 1):]
            project_desc = "".join("".join(x["text"] for x in y) for y in project_desc_lines)
            project_desc = re.sub("^.{,4}[:：]", "", project_desc)
            project_desc_list.append(project_desc)

    return {
        "项目责任": project_desc_list,
        "项目时间": project_time_list,
        "项目名称": project_name_list
    }



with open("../data/Dataset/train_data/train_data.json", encoding="utf-8") as file:
        training_result_dict = json.load(file)

true_result_dict = {}
for resume_id_str, training_resume_dict in training_result_dict.items():
    true_result_dict.update({resume_id_str: training_dict_process(training_resume_dict)})
resume_id_list = sorted(true_result_dict.keys())

company_last_two_list = []
for resume_id_str in resume_id_list:
    true_resume_dict = true_result_dict[resume_id_str]
    for company_name_str in true_resume_dict["工作单位"]:
        last_two_str = company_name_str[-2:]
        if re.match(r'[a-z]+', last_two_str, re.I):
            last_two_str = company_name_str[-4:]
        if company_name_str[-2].isnumeric():
            if company_name_str[-1].isnumeric():
                last_two_str = company_name_str[-5:]
            else:
                last_two_str = "\\d" + company_name_str[-1]
        company_last_two_list.append(last_two_str) 
company_last_two_list = list(set(company_last_two_list))
company_last_two_pattern_str = "|".join(company_last_two_list)
company_last_two_pattern_str = company_last_two_pattern_str.replace(".", "\\.")


bc = BertClient()

data_path = "../data/Dataset/train_data/pdf"

train_x = []
train_y = []

data_path = Path(data_path)
print(">>>> Training...")
print(">> Processing Training PDFs...")
for file_id, filename in enumerate(os.listdir(data_path)):
    print(f"\r {file_id + 1} / 2000", end="")
    pdf = pdfplumber.open(data_path / filename)

    separator_list = get_separator_list(extract_words(pdf.pages[0]),
                                        pdf.pages[0].width)  # 左右分栏的情况
    if separator_list:
        is_left = separator_list[0] <= pdf.pages[0].width / 2
        is_right = separator_list[-1] > (pdf.pages[0].width / 2)
        if is_left:
            separator = separator_list[0]
        elif is_right:
            separator = separator_list[-1]
    else:
        separator = 0
        is_left = 0
        is_right = 0

    word_list = []
    for i, page in enumerate(pdf.pages):
        word_list += extract_words(page, i)

    if is_left:
        ##separator为左模式下提取{filename}的简历信息
        side_part_word_list = [x for x in word_list if x["x1"] < separator]
        line_list = get_main_line_list(
            [x for x in word_list if x["x0"] > separator])
    elif is_right:
        print(f"separator为右模式下提取{filename}的简历信息")
        side_part_word_list = [x for x in word_list if x["x0"] > separator]
        line_list = get_main_line_list(
            [x for x in word_list if x["x1"] < separator])
        if not side_part_word_list:
            print(f"separator为右模式下提取{filename}的简历信息为空")
    else:
        side_part_word_list = [x for x in word_list if x["x1"] < separator]
        line_list = get_main_line_list(
            [x for x in word_list if x["x0"] > separator])

    resume_info = split_main_resume(line_list, True, filename)
    if not resume_info:
        print(f"separator模式下提取{filename}的简历信息发生错误,将separator置0")
        separator = 0
        side_part_word_list = [x for x in word_list if x["x1"] < separator]
        line_list = get_main_line_list(
            [x for x in word_list if x["x0"] > separator])

    resume_info = split_main_resume(line_list, False, filename)
    resume_info["side"] = side_part_word_list

    section_list = [
        section for section in resume_info.keys()
        if resume_info[section] and section != 'side'
    ]
    section_list = sorted(section_list,
                          key=lambda x: (resume_info[x][0][0]['y0']))

    word_list = []
    if resume_info['side']:
        line = sorted(resume_info['side'], key=lambda x: (x['y0'], x['x0']))
        word_list += [[word['text'], 'base'] for word in line]
    for section in section_list:
        line_list = resume_info[section]
        section_word_list = []
        for line in line_list:
            line = sorted(line, key=lambda x: (x['y0'], x['x0']))
            section_word_list += [word['text'] for word in line]
        word_list += [[word, section] for word in section_word_list]
    text_list = [word[0] for word in word_list]
    label_list = [word[1] for word in word_list]
    train_x.append(text_list)
    train_y.append(label_list)

train_x_encoded = []

print("\n")
print(">> Encoding Training Data...")
for word_id, words in enumerate(train_x):
    print(f"\r {word_id + 1} / 2000", end="")
    train_x_encoded.append(bc.encode(words))

# padding x
resume_num = len(train_x_encoded)
# seq_maxlen = max(list(map(len, train_x_encoded)))
seq_maxlen = 190
embedding_dim = 768
train_x_paded = np.zeros((resume_num, seq_maxlen, embedding_dim))

for row, resume in enumerate(train_x_encoded):
    seq_len = len(resume)
    if seq_len <= seq_maxlen:  # right padding
        train_x_paded[row, :seq_len, :] = resume
    else:
        train_x_paded[row, :, :] = resume[-seq_maxlen:]

chunk_tags = ['base', 'edu', 'job', 'project', 'o']

# chunk y

train_y_chunk = []
for label_list in train_y:
    y_chunk = [chunk_tags.index(label) for label in label_list]
    train_y_chunk.append(y_chunk)

# padding y
resume_num = len(train_x_encoded)
# seq_maxlen = max(list(map(len, train_x_encoded)))
seq_maxlen = 190
y_dim = 1
train_y_paded = -np.ones((resume_num, seq_maxlen, y_dim), dtype="int")

for row, resume in enumerate(train_y_chunk):
    seq_len = len(resume)
    if seq_len <= seq_maxlen:
        train_y_paded[row, :seq_len, -1] = resume
    else:
        train_y_paded[row, :, -1] = resume[-seq_maxlen:]


BiRNN_UNITS = 128
model = Sequential()
model.build((None, 190, 768))
model.add(Masking(mask_value=0, input_shape=(190, 768)))
model.add(LSTM(BiRNN_UNITS, return_sequences=True))
model.add(Dropout(rate=0.1))
crf = CRF(5, sparse_target=True)
model.add(crf)
model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])

EPOCHS = 5

print("\n")
print(">> Training Bilstm-crf...")
model.fit(train_x_paded, train_y_paded, batch_size=128, epochs=EPOCHS)

model.save('../user_data/model_data/crf.h5')