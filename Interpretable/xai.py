import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from util import config, file_dir
from graph import Graph
from dataset import HazeData
import numpy as np
from model.EEMD_GNN_GRUA import EEMD_GNN_GRUA

import torch
from torch import nn

torch.set_num_threads(1)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

graph = Graph()
city_num = graph.node_num

batch_size = config['train']['batch_size']
epochs = config['train']['epochs']
hist_len = config['train']['hist_len']
pred_len = config['train']['pred_len']
weight_decay = config['train']['weight_decay']
early_stop = config['train']['early_stop']
lr = config['train']['lr']
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']  # exp_repeat: 10
save_npy = config['experiments']['save_npy']
F = config['train']['F']
criterion = nn.MSELoss()

train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')
in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std
imf_1_mean, imf_1_std = test_data.imf_2_mean, test_data.imf_1_std
#
# 创建模型实例
model = EEMD_GNN_GRUA(hist_len, pred_len, in_dim, city_num, batch_size, device, graph.edge_index, graph.edge_attr,
                      wind_mean, wind_std, F)
# 加载状态字典
model.load_state_dict(torch.load('result/EGGA/model.pth'))
model.to(device)
# 将模型置于评估模式
model.eval()

# 从数据加载器中获取一批数据
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

# 初始化 Captum 的 IntegratedGradients

from captum.attr import FeaturePermutation

feature_perm = FeaturePermutation(model)


# 定义计算特征重要性的函数
def compute_feature_importance(model, dataloader, flag):
    count = 0
    all_attributions = []

    for data in tqdm(dataloader, desc="Computing Feature Importance"):
        # ...您的代码...
        pm25, imf_1, imf_2, imf_3, imf_4, imf_5, imf_6, imf_7, imf_8, imf_9, imf_10, imf_11, imf_12, imf_13, feature, time_arr = data
        pm25 = pm25.to(device)
        imf_1 = imf_1.to(device)
        imf_2 = imf_2.to(device)
        imf_3 = imf_3.to(device)
        imf_4 = imf_4.to(device)
        imf_5 = imf_5.to(device)
        imf_6 = imf_6.to(device)
        imf_7 = imf_7.to(device)
        imf_8 = imf_8.to(device)
        imf_9 = imf_9.to(device)
        imf_10 = imf_10.to(device)
        imf_11 = imf_11.to(device)
        imf_12 = imf_12.to(device)
        imf_13 = imf_13.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        # print(pm25_label.shape)
        imf_1_hist = imf_1[:, :hist_len]  # (16,1,184,1)
        imf_2_hist = imf_2[:, :hist_len]
        imf_3_hist = imf_3[:, :hist_len]
        imf_4_hist = imf_4[:, :hist_len]
        imf_5_hist = imf_5[:, :hist_len]
        imf_6_hist = imf_6[:, :hist_len]
        imf_7_hist = imf_7[:, :hist_len]  # (16,1,184,1)
        imf_8_hist = imf_8[:, :hist_len]
        imf_9_hist = imf_9[:, :hist_len]
        imf_10_hist = imf_10[:, :hist_len]
        imf_11_hist = imf_11[:, :hist_len]
        imf_12_hist = imf_12[:, :hist_len]
        imf_13_hist = imf_13[:, :hist_len]

        attributions = feature_perm.attribute(inputs=(
        imf_1_hist, imf_2_hist, imf_3_hist, imf_4_hist, imf_5_hist, imf_6_hist, imf_7_hist, imf_8_hist, imf_9_hist,
        imf_10_hist, imf_11_hist, imf_12_hist, imf_13_hist, pm25_hist, feature), target=None)
        for i, tensor in enumerate(attributions):
            # 移动到 CPU
            tensor_cpu = tensor.cpu()
            # 或者，转换为 NumPy 数组并保存为 .npy 文件
            numpy_array = tensor_cpu.numpy()

            # np.save(f'xai/all/{flag}/{count}_attribution_tensor_{i}.npy', numpy_array)
        count += 1

    return None


# 对每个数据集计算特征重要性
# train_attributions = compute_feature_importance(model, train_loader,'train')
# print('train')
val_attributions = compute_feature_importance(model, val_loader, 'val')
print('val')
# test_attributions = compute_feature_importance(model, test_loader,'test')
# print('test')
