import argparse
import os
import torch

from dataset.dataset import load_data, normalize_features_and_labels, build_dynamic_graphs, split_data, \
    create_data_loaders
from evaluate import evaluate_model, visualize_predictions
from models.model import GC_TCN, GCN_LSTM, GCN_GRU
from train import run_training_loop
from utils.utils import seed_torch, load_dataloader, save_data_loaders


def main(args):
    # -------------------------------------------- 1. seed -------------------------------------------------------------
    seed_torch(args.seed)

    # -------------------------------------------- 2. device -----------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------- 3. data -------------------------------------------------------------
    if os.path.exists(args.loader_path):
        print(f"加载现有的dataloader从 {args.loader_path}")
        train_loader, val_loader, test_loader = load_dataloader(args.loader_path)
    else:
        print("创建新的dataloader并保存...")
        df = load_data(args.data_path)  # 读取数据
        data_normalized, labels_normalized = normalize_features_and_labels(df, args.target)  # 数据标准化
        graphs = build_dynamic_graphs(data_normalized, labels_normalized, args.seq_len, args.step,
                                      args.threshold)  # 生成图
        print(f'窗口数: {len(graphs)}，Data[0]: {graphs[0]}')
        train_data, val_data, test_data = split_data(graphs)  # 数据划分
        train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data,
                                                                    args.batch_size)  # 创建DataLoader
        save_data_loaders(args.loader_path, (train_loader, val_loader, test_loader))  # 保存DataLoader

    # -------------------------------------------- 4. model ------------------------------------------------------------
    if args.model == 'gc-tcn':
        model = GC_TCN(args.num_node_features, args.gcn_hidden_dim, args.tcn_num_channels, args.output_size,
                       args.tcn_kernel_size, args.dropout).to(device)
    elif args.model == 'gcn-lstm':
        model = GCN_LSTM(args.num_node_features, args.gcn_hidden_dim, args.lstm_hidden_dim, args.output_size,
                         args.num_lstm_layers, args.dropout).to(device)
    elif args.model == 'gcn-gru':
        model = GCN_GRU(args.num_node_features, args.gcn_hidden_dim, args.gru_hidden_dim, args.output_size,
                        args.num_gru_layers, args.dropout).to(device)
    else:
        raise ValueError(f"未知模型类型: {args.model}")

    # -------------------------------------------- 5. train ------------------------------------------------------------
    criterion = torch.nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化器

    model_weights_path = os.path.join(args.model_path, args.weight_name)  # 模型权重路径 路径/文件名
    if os.path.isfile(model_weights_path):  # 如果存在预训练权重，则加载
        print(f"加载现有的权重从 {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path))
    else:
        print("未找到预训练权重，开始训练模型")
        run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, args.num_epochs,
                          model_weights_path)

    # -------------------------------------------- 6. evaluate ---------------------------------------------------------
    labels, predictions, lower_bound, upper_bound = evaluate_model(model, test_loader, device,
                                                                   args.mc_samples, args.prediction_interval)
    # -------------------------------------------- 7. visualization ----------------------------------------------------
    visualize_predictions(labels, predictions, lower_bound, upper_bound, args.prediction_interval, args.savefig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GC-TCN')

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    # --------------------------------------------  dataset settings ---------------------------------------------------
    parser.add_argument('--data_path', type=str, default='./dataset/pre_sh_carbon_1700.xlsx', help='which data to use')
    parser.add_argument('--seq_len', type=int, default=30, help='窗口大小')
    parser.add_argument('--step', type=int, default=1, help='步长')
    parser.add_argument('--threshold', type=int, default=0.6, help='相似性阈值')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--target', type=str, default=0, help='标签列')
    parser.add_argument('--loader_path', type=str, default='./dataset/dataloader/d-sh-dataloader.pth', help='标签列')
    # --------------------------------------------  model settings -----------------------------------------------------
    parser.add_argument('--gcn_hidden_dim', type=int, default=64, help='GCN层隐藏层维度')
    # LSTM
    parser.add_argument('--lstm_hidden_dim', type=int, default=64, help='LSTM层隐藏层维度')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='LSTM层数量')
    # GRU
    parser.add_argument('--gru_hidden_dim', type=int, default=64, help='GRU层隐藏层维度')
    parser.add_argument('--num_gru_layers', type=int, default=2, help='GRU层数量')
    # TCN
    parser.add_argument('--tcn_num_channels', type=int, default=[16, 8, 4], help='TCN层通道数量')
    parser.add_argument('--tcn_kernel_size', type=int, default=2, help='TCN层卷积核大小')
    # --------------------------------------------  train settings -----------------------------------------------------
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--num_node_features', type=int, default=30, help='输入维度')
    parser.add_argument('--output_size', type=int, default=1, help='输出维度')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--dropout', type=int, default=0.5, help='dropout rate')
    parser.add_argument('--model', type=str, default='gcn-lstm', choices=['gc-tcn', 'gcn-lstm', 'gcn-gru'])
    parser.add_argument('--model_path', type=str, default="./checkpoints/gcn-lstm/init/", help='保存权重文件的路径')
    parser.add_argument('--weight_name', type=str, default='d-sh-model.pth', help='权重文件名')
    # --------------------------------------------  evaluate settings --------------------------------------------------
    parser.add_argument('--mc_samples', type=int, default=100, help='MC Dropout样本数量')
    parser.add_argument('--prediction_interval', type=bool, default=True, help='是否进行区间预测')
    parser.add_argument('--savefig_path', type=str, default='./vis/GL-sh.png', help='是否可视化预测结果')

    config = parser.parse_args()

    main(config)
