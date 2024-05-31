"""
    训练部分函数
"""
import time

import numpy as np
import torch
import wandb
from sklearn.metrics import r2_score

from torch.utils.tensorboard import SummaryWriter
from utils.util import plot_training_metrics, directional_accuracy


def val_evaluate(model, dataloader, criterion, device):
    """验证评估"""
    model.eval()
    total_loss = 0
    all_preds = []  # 存储所有预测值
    all_labels = []  # 存储所有标签值
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)    # todo
            loss = criterion(out, data.y)
            total_loss += loss.item()
            all_preds.append(out.view(-1).cpu().numpy())
            all_labels.append(data.y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.concatenate(all_preds), np.concatenate(all_labels)  # 返回验证平均损失，所有预测值，所有标签值


def run_training_tensorboard(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs, model_weights_path):
    """
        训练循环
        保存验证损失最低的模型
    """
    # 初始化TensorBoard writer
    writer = SummaryWriter()
    print(model)
    print('#---------------------------------------------------------------------------------------------------#')
    print('#--------------------------------------------开始训练模型--------------------------------------------#')

    # 用于收集指标的列表
    train_losses = []
    val_losses = []
    train_r2s = []
    val_r2s = []
    # train_das = []  # 存储训练集的DA
    # val_das = []  # 存储验证集的DA

    train_predictions = []
    train_labels = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()  # 开始时间
        processed_data = 0  # 已处理数据计数

        for batch_idx, data in enumerate(train_loader, start=1):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)    # todo
            train_predictions.append(out.detach().cpu().numpy())
            train_labels.append(data.y.cpu().numpy())
            loss = criterion(out, data.y)
            # loss = criterion(out, data.y.squeeze())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # 打印进度条
            processed_data += len(data)
            percent = (batch_idx / len(train_loader)) * 100
            filled_length = int(30 * batch_idx // len(train_loader))
            bar = '=' * filled_length + ' ' * (30 - filled_length)
            print(f'\rEpoch {epoch + 1}/{num_epochs} [{processed_data}/{len(train_loader.dataset)}] [{bar}] - '
                  f'{time.time() - start_time:.0f}s {time.time() - start_time:.0f}us/sample - loss: {loss.item():.4f}',
                  end='')

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_preds, val_labels = val_evaluate(model, val_loader, criterion, device)

        # 计算DA评分和R2
        # train_da = directional_accuracy(np.concatenate(train_labels), np.concatenate(train_predictions))
        # val_da = directional_accuracy(val_labels, val_preds)
        train_r2 = r2_score(np.concatenate(train_labels), np.concatenate(train_predictions))
        val_r2 = r2_score(val_labels, val_preds)

        # print(f"Epoch [{epoch + 1}/{num_epochs}] train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f} | "
        #       f"train_r2: {train_r2:.4f}, val_r2: {val_r2:.4f}")

        # 在每个epoch结束时打印结果
        epoch_time = time.time() - start_time  # 这是整个epoch的时间，单位为秒
        time_per_sample = (epoch_time / len(train_loader.dataset)) * 1000  # 换算成毫秒
        print(
            f'\rEpoch {epoch + 1}/{num_epochs} [{len(train_loader.dataset)}/{len(train_loader.dataset)}] [{"=" * 30}] - '
            f'{time_per_sample:.0f}s {time_per_sample:.0f}us/sample - loss: {avg_train_loss:.4f} - r2: {train_r2:.4f} '
            f'- val_loss: {val_loss:.4f} - val_r2: {val_r2:.4f}')

        # 收集指标
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        # train_das.append(train_da)
        # val_das.append(val_da)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)

        # 向TensorBoard写入数据
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        # writer.add_scalar('DA/Train', train_da, epoch)
        # writer.add_scalar('DA/Val', val_da, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('R2/Val', val_r2, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_weights_path)
            print(f"[INFO] Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"save with best val loss: {best_val_loss:.4f}")

    # 结束训练后关闭writer
    writer.close()

    # 调用绘图函数
    # plot_training_metrics(train_losses, val_losses, train_r2s, val_r2s)

    return model


def run_training_wandb(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, model_weights_path):
    """
        训练循环
        保存验证损失最低的模型
    """
    print(model)
    print('#---------------------------------------------------------------------------------------------------#')
    print('#--------------------------------------------开始训练模型--------------------------------------------#')

    train_predictions = []
    train_labels = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        start_time = time.time()  # 开始时间
        processed_data = 0  # 已处理数据计数

        for batch_idx, data in enumerate(train_loader, start=1):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.target_node_index)
            train_predictions.append(out.detach().cpu().numpy())
            train_labels.append(data.y.cpu().numpy())
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # 打印进度条
            processed_data += len(data)
            percent = (batch_idx / len(train_loader)) * 100
            filled_length = int(30 * batch_idx // len(train_loader))
            bar = '=' * filled_length + '-' * (30 - filled_length)
            print(f'\rEpoch {epoch + 1}/{num_epochs} [{processed_data}/{len(train_loader.dataset)}] [{bar}] - '
                  f'{time.time() - start_time:.0f}s {time.time() - start_time:.0f}us/sample - loss: {loss.item():.4f}',
                  end='')

        avg_train_loss = total_train_loss / len(train_loader)   # 训练损失
        val_loss, val_preds, val_labels = val_evaluate(model, val_loader, criterion, device)    # 验证损失
        train_r2 = r2_score(np.concatenate(train_labels), np.concatenate(train_predictions))    # 训练R2
        val_r2 = r2_score(val_labels, val_preds)    # 验证R2

        # 使用wandb记录损失和R2
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "train_r2": train_r2,
            "val_r2": val_r2
        })

        # 在每个epoch结束时打印结果
        epoch_time = time.time() - start_time  # 这是整个epoch的时间，单位为秒
        time_per_sample = (epoch_time / len(train_loader.dataset)) * 1000  # 换算成毫秒
        print(
            f'\rEpoch {epoch + 1}/{num_epochs} [{len(train_loader.dataset)}/{len(train_loader.dataset)}] [{"=" * 30}] - '
            f'{time_per_sample:.0f}s {time_per_sample:.0f}us/sample - loss: {avg_train_loss:.4f} - r2: {train_r2:.4f} '
            f'- val_loss: {val_loss:.4f} - val_r2: {val_r2:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_weights_path)
            print(f"[INFO] Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"save with best val loss: {best_val_loss:.4f}")

    return model