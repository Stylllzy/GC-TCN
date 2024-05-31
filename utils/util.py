"""
    工具函数文件
"""
import itertools
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
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


def set_device():
    """设置设备"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device --> [{torch.cuda.get_device_name(device)}]")
    print('#---------------------------------------------------------------------------------------------------#')
    return device


def dtw_to_similarity(dtw_distance, alpha=1):
    """将两个序列之间的 DTW 距离转换为相似度（0，1）之间"""
    return np.exp(-dtw_distance / alpha)


def calc_pearson(x, y):
    """计算皮尔逊相关系数."""
    # 使用 scipy 的 pearsonr 函数计算sub_data中每列之间的相关系数
    corr, p = pearsonr(np.array(x), np.array(y))
    return corr


def calc_pearson_matrix(data):
    """计算数据矩阵的皮尔逊相关系数矩阵"""
    # 减去均值
    data_mean = data - np.mean(data, axis=0)
    # 计算标准差，避免除以零
    data_std_dev = np.std(data, axis=0)
    data_std_dev[data_std_dev == 0] = 1
    # 计算标准化的数据矩阵
    data_std = data_mean / data_std_dev
    # 计算相关系数矩阵
    correlation_matrix = np.dot(data_std.T, data_std) / data.shape[0]
    return correlation_matrix


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


def sequential_split(data_list, lengths):
    """将列表按照时间顺序拆分为由长度指定的非重叠新列表。"""
    if sum(lengths) != len(data_list):
        raise ValueError("Sum of input lengths does not equal the length of the input list!")

    # Sequential split
    train = data_list[:lengths[0]]
    val = data_list[lengths[0]:lengths[0] + lengths[1]]
    test = data_list[lengths[0] + lengths[1]:]

    return train, val, test


def save_data_loaders(filepath, loaders):
    """保存dataloader"""
    torch.save(loaders, filepath)


def load_dataloader(filepath):
    """加载dataloader"""
    train_loader, val_loader, test_loader = torch.load(filepath)
    return train_loader, val_loader, test_loader


def directional_accuracy(y_true, y_pred):
    """
    计算方向准确性，y_true是实际值，y_pred是预测值
    我们需要比较连续的预测值和实际值之间的变化趋势
    """
    n = len(y_true)
    correct_directions = 0
    for i in range(1, n):
        # 实际值的变化趋势
        actual_direction = np.sign(y_true[i] - y_true[i - 1])
        # 预测值的变化趋势
        predicted_direction = np.sign(y_pred[i] - y_pred[i - 1])
        # 如果变化趋势相同（都是正或都是负）则视为正确
        if actual_direction == predicted_direction:
            correct_directions += 1
    return correct_directions / (n - 1)


def compute_metrics(labels, predictions):
    """计算评估指标."""
    predictions = np.array(predictions)
    labels = np.array(labels)

    rmse = np.sqrt(mean_squared_error(labels, predictions))  # 均方根误差
    mae = mean_absolute_error(labels, predictions)  # 平均绝对误差
    r2 = r2_score(labels, predictions)  # R2
    mape = mean_absolute_percentage_error(labels, predictions)  # 平均绝对百分比误差
    smape = 2.0 * np.mean(np.abs(predictions - labels) / (np.abs(predictions) + np.abs(labels))) * 100  # 对称平均绝对百分比误差
    da = directional_accuracy(labels, predictions)  # 方向准确性

    return mae, rmse, mape, r2, da


def moving_average(data, window_size=5):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_training_metrics(epochs, train_losses, val_losses, train_r2s, val_r2s):
    """ 绘制训练过程中的损失和 R2 准确度图表 """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.labelsize'] = 18  # x轴刻度的字体大小
    plt.rcParams['ytick.labelsize'] = 18  # y轴刻度的字体大小
    # Set font sizes
    title_fontsize = 22
    label_fontsize = 20
    legend_fontsize = 24

    plt.figure(figsize=(12, 5))

    # 绘制损失图表
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', c='red')
    plt.plot(epochs, val_losses, label='Validation Loss', c='blue')
    plt.title('Loss over Epochs', fontsize=title_fontsize)
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.grid(True)
    plt.legend(fontsize=legend_fontsize)

    # 绘制R2准确率图表
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_r2s, label='Train R2 Score', c='red')
    plt.plot(epochs, val_r2s, label='Validation R2 Score', c='blue')
    plt.title('R2 Score over Epochs', fontsize=title_fontsize)
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('R2 Score', fontsize=label_fontsize)
    plt.grid(True)
    plt.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig('loss图.pdf', bbox_inches='tight', dpi=600, pad_inches=0.0)
    plt.show()
