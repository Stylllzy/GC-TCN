"""
    工具函数文件
"""
import itertools
import random

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
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


def save_data_loaders(filepath, loaders):
    """保存dataloader"""
    torch.save(loaders, filepath)


def load_dataloader(filepath):
    """加载dataloader"""
    train_loader, val_loader, test_loader = torch.load(filepath)
    return train_loader, val_loader, test_loader


def compute_metrics(labels, predictions):
    """计算评估指标."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    rmse = np.sqrt(mean_squared_error(labels, predictions))  # 均方根误差
    mae = mean_absolute_error(labels, predictions)  # 平均绝对误差
    r2 = r2_score(labels, predictions)  # R2
    mape = mean_absolute_percentage_error(labels, predictions)  # 平均绝对百分比误差
    return mae, rmse, mape, r2


def moving_average(data, window_size=5):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')