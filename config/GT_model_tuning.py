import numpy as np
import optuna
import torch
from tqdm import tqdm

from models.model import GC_TCN, GCN_LSTM
from utils.util import seed_torch, load_dataloader, compute_metrics

seed_torch(42)  # 设置随机种子

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练模型
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss  # 返回训练平均损失


# 评估模型
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []  # 存储所有预测值
    all_labels = []  # 存储所有标签值
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            all_preds.append(out.view(-1).cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_labels)  # 返回验证平均损失，所有预测值，所有标签值


# 1. 定义目标函数
def objective(trial):
    # 为超参数提供建议
    num_node_features = 30
    gcn_hidden_dim = trial.suggest_int("gcn_hidden_dim", 32, 128, step=16)

    lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 16, 128, step=16)
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)

    # tcn_num_channels = [64, 32, 16]

    # kernel_size = trial.suggest_categorical("kernel_size", [2, 3, 5])

    dropout = trial.suggest_float("dropout", 0.2, 0.5, step=0.1)
    learning_rate = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    num_classes = 1

    # 加载dataloaders
    train_loader, val_loader, test_loader = load_dataloader(
        '../dataset/water/RJY-dataloader.pth')

    # 初始化模型
    # model = GC_TCN(num_node_features, gcn_hidden_dim, tcn_num_channels, num_classes, kernel_size, dropout).to(
    #     device)
    model = GCN_LSTM(num_node_features, gcn_hidden_dim, lstm_hidden_dim, num_classes,
                     num_lstm_layers, dropout).to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    val_loss = 0.0
    num_epochs = 100
    preds = []
    labels = []

    # 训练模型
    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, preds, labels = evaluate(model, val_loader, criterion, device)

    _, _, _, r2, _ = compute_metrics(labels, preds)

    return r2  # 我们希望最大化 R2 分数


# 创建一个 Optuna 学习对象并开始优化
study = optuna.create_study(study_name='GCLSTM-水质-R2', direction="maximize",
                            storage='sqlite:///db.sqlite3')  # 我们希望最小化验证损失
study.optimize(objective, n_trials=50)

# 输出最佳超参数
print(f"The best hyperparameters are {study.best_params}\nWith a validation loss of {study.best_value}")

# 水质


# optuna-dashboard sqlite:///db1.sqlite3

# study_name = 'GCTCN-上海-R2'
# The best hyperparameters are {'gcn_hidden_dim': 112, 'kernel_size': 3, 'dropout': 0.2, 'lr': 0.0013114035403734698}
# With a validation loss of 0.9731086353331234

# study_name = 'GCTCN-广东-R2'
# The best hyperparameters are {'gcn_hidden_dim': 96, 'kernel_size': 5, 'dropout': 0.30000000000000004, 'lr': 5.2826234249947204e-05}
# With a validation loss of 0.891756127792374

# study_name = 'GCTCN-湖北-R2'
# The best hyperparameters are {'gcn_hidden_dim': 96, 'kernel_size': 2, 'dropout': 0.4, 'lr': 0.000665601666979068}
# With a validation loss of 0.9712340334221333