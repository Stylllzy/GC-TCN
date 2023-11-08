"""
    工具函数文件
"""
import itertools
import random

import numpy as np
import torch
from scipy.stats import pearsonr
from tslearn.metrics import dtw as ts_dtw


def seed_torch(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dtw_to_similarity(dtw_distance, alpha=1):
    """将两个序列之间的 DTW 距离转换为相似度（0，1）之间"""
    return np.exp(-dtw_distance / alpha)


def calc_pearson(x, y):
    """计算皮尔逊相关系数."""
    # 使用 scipy 的 pearsonr 函数计算sub_data中每列之间的相关系数
    corr, p = pearsonr(np.array(x), np.array(y))
    return corr


def calc_dtw(x, y):
    """计算两个序列之间的 DTW 距离."""
    distance = ts_dtw(x, y)
    similarity = dtw_to_similarity(distance)
    return distance, similarity


def custom_random_split(data_list, lengths):
    """将列表随机拆分为由长度指定的非重叠新列表。"""
    if sum(lengths) != len(data_list):
        raise ValueError("Sum of input lengths does not equal the length of the input list!")

    # Shuffle the data
    indices = list(range(len(data_list)))
    random.shuffle(indices)

    # Split according to the lengths
    return [[data_list[i] for i in indices[offset - length: offset]] for offset, length in
            zip(itertools.accumulate(lengths), lengths)]