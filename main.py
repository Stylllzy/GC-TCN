import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from config.config import default_target, default_loader_path, default_gcn_hidden_dim, default_lstm_hidden_dim, \
    default_num_lstm_layers, default_gru_hidden_dim, default_num_gru_layers, default_tcn_num_channels, \
    default_tcn_kernel_size, default_lr, default_num_epochs, default_dropout, default_model, default_model_path, \
    default_weight_name, default_savefig_path, default_tcn_dilation_size, default_data_path
from dataset.dataset import load_data, normalize_features_and_labels, split_data, \
    create_data_loaders, build_dynamic_graphs
from evaluate import evaluate_model, visualize_predictions, visualize_predictions_interval
from models.model import GC_TCN, GCN_LSTM, GCN_GRU, GCN_FC
from train import run_training_wandb, run_training_tensorboard
from utils.util import seed_torch, load_dataloader, save_data_loaders, set_device


# import wandb


def main(args):
    print(f"当前超参数 lr: {args.lr}---GCN隐藏层维度: {args.gcn_hidden_dim}---扩张大小: {args.dilation_size}")
    # -------------------------------------------- 0. 初始化wandb -------------------------------------------------------
    # wandb.init(project="DGCPN", name="water", config=args)

    # -------------------------------------------- 1. seed -------------------------------------------------------------
    seed_torch(args.seed)

    # -------------------------------------------- 2. device -----------------------------------------------------------
    device = set_device()

    # -------------------------------------------- 3. data -------------------------------------------------------------
    df, train_size, val_size, test_size, train_times, val_times, test_times, feature_name = load_data(
        args.data_path)  # 读取数据
    print(f"文件路径: {args.data_path}, \n"
          f"训练集大小: {train_size}, 训练集时间: {train_times[0]} - {train_times[-1]}, \n"
          f"验证集大小: {val_size}, 验证集时间: {val_times[0]} - {val_times[-1]}, \n"
          f"测试集大小: {test_size}, 测试集时间: {test_times[0]} - {test_times[-1]}")
    data_normalized, labels_normalized, _, scaler_labels = normalize_features_and_labels(df,
                                                                                         args.target)  # 数据标准化
    if os.path.exists(args.loader_path):
        print(f"加载现有的dataloader从 {args.loader_path}")
        print('#---------------------------------------------------------------------------------------------------#')
        train_loader, val_loader, test_loader = load_dataloader(args.loader_path)
        print(f"窗口数: {len(train_loader.dataset)}, Data[0]: {train_loader.dataset[0]}")
    else:
        print("创建新的dataloader并保存...")
        print('#---------------------------------------------------------------------------------------------------#')
        graphs = build_dynamic_graphs(data_normalized, labels_normalized, device, args.seq_len, args.step,
                                      args.threshold)  # 生成图
        print(f'窗口数: {len(graphs)}，Data[0]: {graphs[0]}')
        train_data, val_data, test_data = split_data(graphs)  # 数据划分
        train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data,
                                                                    args.batch_size)  # 创建DataLoader
        save_data_loaders(args.loader_path, (train_loader, val_loader, test_loader))  # 保存DataLoader

    # -------------------------------------------- 4. model ------------------------------------------------------------
    if args.model == 'gc-tcn':
        model = GC_TCN(args.num_node_features, args.gcn_hidden_dim, args.tcn_num_channels, args.output_size,
                       args.tcn_kernel_size, args.dropout, args.dilation_size).to(device)
    elif args.model == 'gc-lstm':
        model = GCN_LSTM(args.num_node_features, args.gcn_hidden_dim, args.lstm_hidden_dim, args.output_size,
                         args.num_lstm_layers, args.dropout).to(device)
    elif args.model == 'gc-gru':
        model = GCN_GRU(args.num_node_features, args.gcn_hidden_dim, args.gru_hidden_dim, args.output_size,
                        args.num_gru_layers, args.dropout).to(device)
    elif args.model == 'gc-fc':
        model = GCN_FC(args.num_node_features, args.gcn_hidden_dim, args.output_size).to(device)
    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # -------------------------------------------- 5. train ------------------------------------------------------------
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)  # 学习率衰减

    model_weights_path = os.path.join(args.model_path, args.weight_name)  # 模型权重路径 路径/文件名
    if os.path.isfile(model_weights_path):  # 如果存在预训练权重，则加载
        print(f"加载现有的权重从 {model_weights_path}")
        print('#---------------------------------------------------------------------------------------------------#')
        model.load_state_dict(torch.load(model_weights_path))
    else:
        print("未找到预训练权重，开始训练模型")
        print('#---------------------------------------------------------------------------------------------------#')
        run_training_tensorboard(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
                                 args.num_epochs, model_weights_path)
        # run_training_wandb(model, train_loader, val_loader, optimizer, criterion, device, args.num_epochs,
        #                    model_weights_path)

    # ------------------------------------ 6. evaluate & visualization -------------------------------------------------
    if args.prediction_interval:
        labels, predictions, lower_bound, upper_bound = evaluate_model(model, test_loader, device,
                                                                       args.mc_samples, args.prediction_interval,
                                                                       scaler_labels)
        visualize_predictions_interval(test_times, labels, predictions, lower_bound, upper_bound, args.savefig_path)
    else:
        labels, predictions = evaluate_model(model, test_loader, device,
                                             args.mc_samples, args.prediction_interval, scaler_labels)
        # visualize_predictions(labels, predictions, args.savefig_path)

        # 保存预测结果到CSV
        # predictions_df = pd.DataFrame({
        #     'Predictions': predictions,
        #     'True_Values': labels
        # })
        # # 保存到CSV
        # predictions_df.to_csv('DGTCN-water_predictions.csv', index=False)

    # ------------------------------------ 7. Captum Feature Permutation 可解释 -----------------------------------------
    # todo

if __name__ == '__main__':
    start_time = time.time()  # 开始计时
    parser = argparse.ArgumentParser(description='GC-TCN')

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    # --------------------------------------------  dataset settings ---------------------------------------------------
    parser.add_argument('--data_path', type=str, default=default_data_path, help='数据集源文件')
    parser.add_argument('--seq_len', type=int, default=30, help='窗口大小')
    parser.add_argument('--step', type=int, default=1, help='步长')
    parser.add_argument('--threshold', type=int, default=0.7, help='相似性阈值')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--target', type=str, default=default_target, help='标签列')
    parser.add_argument('--loader_path', type=str, default=default_loader_path, help='加载dataloader')
    # --------------------------------------------  model settings -----------------------------------------------------
    parser.add_argument('--gcn_hidden_dim', type=int, default=default_gcn_hidden_dim, help='GCN层隐藏层维度')
    # LSTM
    parser.add_argument('--lstm_hidden_dim', type=int, default=default_lstm_hidden_dim, help='LSTM层隐藏层维度')
    parser.add_argument('--num_lstm_layers', type=int, default=default_num_lstm_layers, help='LSTM层数量')
    # GRU
    parser.add_argument('--gru_hidden_dim', type=int, default=default_gru_hidden_dim, help='GRU层隐藏层维度')
    parser.add_argument('--num_gru_layers', type=int, default=default_num_gru_layers, help='GRU层数量')
    # TCN
    parser.add_argument('--tcn_num_channels', type=int, default=default_tcn_num_channels, help='TCN层通道数量')
    parser.add_argument('--tcn_kernel_size', type=int, default=default_tcn_kernel_size, help='TCN层卷积核大小')
    parser.add_argument('--dilation_size', type=int, default=default_tcn_dilation_size, help='TCN层膨胀大小')
    # --------------------------------------------  train settings -----------------------------------------------------
    parser.add_argument('--lr', type=int, default=default_lr, help='learning rate')
    parser.add_argument('--num_node_features', type=int, default=30, help='输入维度')
    parser.add_argument('--output_size', type=int, default=1, help='输出维度')
    parser.add_argument('--num_epochs', type=int, default=default_num_epochs, help='训练轮数')
    parser.add_argument('--dropout', type=int, default=default_dropout, help='dropout rate')
    parser.add_argument('--model', type=str, default=default_model, choices=['gc-fc', 'gc-tcn', 'gc-lstm', 'gc-gru'])
    parser.add_argument('--model_path', type=str, default=default_model_path, help='保存权重文件的路径')
    parser.add_argument('--weight_name', type=str, default=default_weight_name, help='权重文件名')
    # --------------------------------------------  evaluate settings --------------------------------------------------
    parser.add_argument('--mc_samples', type=int, default=100, help='MC Dropout样本数量')
    parser.add_argument('--prediction_interval', type=bool, default=False, help='是否进行区间预测')
    parser.add_argument('--savefig_path', type=str, default=default_savefig_path, help='是否可视化预测结果')

    config = parser.parse_args()

    main(config)
    # wandb.finish()

    # 结束计时，并计算总时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"总时间: {total_time // 60:.0f} 分钟 {total_time % 60:.0f} 秒")