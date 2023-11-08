import argparse
import os
import torch
import numpy as np

from dataset.dataset import load_data, normalize_features_and_labels, build_dynamic_graphs, split_data, \
    create_data_loaders, save_data_loaders
from utils.utils import seed_torch


def main(args):
    # 1. seed
    seed_torch(args.seed)

    # 2. device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. data
    df = load_data(args.data_path)  # 读取数据
    data_normalized, labels_normalized = normalize_features_and_labels(df, args.target)  # 数据标准化
    graphs = build_dynamic_graphs(data_normalized, labels_normalized, args.seq_len, args.step, args.threshold)  # 生成图
    print(f'窗口数: {len(graphs)}，Data[0]: {graphs[0]}')
    train_data, val_data, test_data = split_data(graphs)  # 数据划分  80%-train 10%-val 10%-test
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data,
                                                                args.batch_size)  # 创建DataLoader
    save_data_loaders(args.loader_path, (train_loader, val_loader, test_loader))  # 保存DataLoader


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GC-TCN')

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

    ### -------  dataset settings --------------
    parser.add_argument('--data_path', type=str, default='./dataset/pre_sh_carbon_1700.xlsx', help='which data to use')
    parser.add_argument('--seq_len', type=int, default=30, help='窗口大小')
    parser.add_argument('--step', type=int, default=1, help='步长')
    parser.add_argument('--threshold', type=int, default=0.6, help='相似性阈值')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--target', type=str, default=0, help='标签列')
    parser.add_argument('--loader_path', type=str, default='./dataset/dataloader/d-sh-dataloader.pth', help='标签列')

    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden units')
    parser.add_argument('--output_size', type=int, default=1, help='输出维度')
    parser.add_argument('--num_epochs', type=int, default=200, help='num epochs')
    parser.add_argument('--num_layers', type=int, default=2, help='num layer dim')
    parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')
    parser.add_argument('--model', type=str, default='gc-tcn', choices=['gc-tcn'])
    parser.add_argument('--model_path', type=str, default="./save_dict/", help='save_model')

    config = parser.parse_args()

    main(config)
