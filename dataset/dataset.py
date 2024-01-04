"""
    数据处理相关的函数
"""
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from scipy import stats
from torch_geometric.data import Data

from utils.util import calc_dtw, custom_random_split, sequential_split


def load_data(filepath):
    """加载数据"""
    # data = pd.read_excel(filepath, sheet_name=0, parse_dates=['date'], index_col='date')
    data = pd.read_csv(filepath, encoding='ISO-8859-1', parse_dates=['date'], index_col='date').dropna()
    total_size = len(data)
    test_size = int(total_size * 0.1)   # 测试集占总数据的比例
    test_times = data.index[-test_size:]
    return data, test_times


def normalize_features_and_labels(data, label):
    """
        对数变换标签列
        选取特征和标签
        标准化
    """
    # scaler_data = MinMaxScaler(feature_range=(0, 1))
    # scaler_labels = MinMaxScaler(feature_range=(0, 1))
    scaler_data = StandardScaler()
    scaler_labels = StandardScaler()

    # 对标签列进行差分
    # labels_diff = data[label].diff().dropna()
    # # 删除特征数据的第一行以匹配标签长度
    # data_adjusted = data.iloc[1:, :]

    # 选择特征和标签
    data_selected = data.values
    labels = data[label].values

    data_normalized = scaler_data.fit_transform(data_selected)
    labels_normalized = scaler_labels.fit_transform(labels.reshape(-1, 1))

    return data_normalized, labels_normalized, scaler_data, scaler_labels


def build_edges(data, threshold, method='dtw'):
    """通过计算相似性确立边"""
    num_nodes = data.shape[1]
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)
                    distance, similarity = calc_dtw(data[:, i], data[:, j])
                if similarity > threshold:
                    edge_index.append((i, j))
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def build_dynamic_graphs(data, labels, window_size=30, step_size=1, threshold=0.5, target_node_index=0):
    """
        滑窗处理
        处理好的x,y,edge_index封装到Data对象
    """
    graphs = []
    for i in tqdm(range(0, data.shape[0] - window_size + 1, step_size)):
        window_data = data[i:i + window_size]
        edge_index = build_edges(window_data, threshold)
        x = torch.tensor(window_data, dtype=torch.float).t()  # Transpose to have [num_nodes, window_size]
        # y = torch.tensor([labels[i + window_size - 1]], dtype=torch.float).view(1, 1)
        y = torch.from_numpy(np.array(labels[i + window_size - 1])).float().view(1, 1)

        # y = torch.tensor(labels[i + window_size - 1], dtype=torch.float).view(1, -1)      # 针对多输出

        graph_data = Data(x=x, edge_index=edge_index, y=y, target_node_index=torch.tensor([target_node_index]))
        graphs.append(graph_data)
    return graphs


def split_data(graphs):
    """按时间顺序分割数据"""
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    test_size = len(graphs) - train_size - val_size
    return sequential_split(graphs, [train_size, val_size, test_size])


def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """创建dataloader"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
