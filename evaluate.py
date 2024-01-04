"""
    评估部分函数
"""
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.util import compute_metrics, moving_average


def mc_dropout_predict(model, dataloader, device, num_samples=100):
    """
        蒙特卡洛 (MC) Dropout：
            如果在训练模型时使用了dropout，那么在预测时可以继续使用dropout来进行多次预测（例如100次）。这样，对于每个时间点，您都会得到一个预测值的分布。
            计算这些预测值的均值和标准差，以获得点预测和预测区间。
            这种方法基于的思想是，dropout可以模拟模型的集成，从而为预测提供不确定性估计。
    """
    model.train()  # 确保模型处于训练模式，启用 Dropout

    all_mean_predictions = []  # 所有时间点的预测均值
    all_std_predictions = []  # 所有时间点的预测标准差
    all_labels = []  # 所有时间点的真实值

    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_predictions = []

        batch = batch.to(device)

        for _ in range(num_samples):  # Nested progress bar
            out = model(batch).detach().cpu().numpy()
            batch_predictions.append(out)

        batch_mean = np.mean(batch_predictions, axis=0)
        batch_std = np.std(batch_predictions, axis=0)

        all_mean_predictions.extend(batch_mean)
        all_std_predictions.extend(batch_std)
        all_labels.extend(batch.y.cpu().numpy())  # 收集真实标签

    return np.array(all_mean_predictions), np.array(all_std_predictions), np.array(all_labels)


def point_predict(model, dataloader, device):
    """标准点预测"""
    model.eval()  # 确保模型处于评估模式，禁用 Dropout
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch).detach().cpu().numpy()
            all_predictions.extend(out)
            all_labels.extend(batch.y.cpu().numpy())

    return np.array(all_predictions).squeeze(), None, np.array(all_labels).squeeze()


def evaluate_model(model, dataloader, device, num_samples=100, prediction_interval=False, scaler_labels=None):
    """
        评估模型
        针对点预测和预测区间进行评估
    """
    print('#---------------------------------------------------------------------------------------------------#')
    print('#--------------------------------------------开始评估模型--------------------------------------------#')
    predictions, stds, labels = mc_dropout_predict(model, dataloader, device, num_samples) if prediction_interval \
        else point_predict(model, dataloader, device)

    z_score = 1.96
    if prediction_interval:
        lower_bound = predictions - z_score * stds
        upper_bound = predictions + z_score * stds

        all_labels = np.array(labels).squeeze()  # 将（168，1） -> （168，）
        mean_predictions = predictions.squeeze()
        lower_bound = lower_bound.squeeze()
        upper_bound = upper_bound.squeeze()

        coverage = np.mean((labels >= lower_bound) & (labels <= upper_bound))
        print(f"95% Prediction Interval Coverage: {coverage * 100:.2f}%")

        # 反标准化
        if scaler_labels is not None:
            all_labels = scaler_labels.inverse_transform(all_labels.reshape(-1, 1)).squeeze()
            mean_predictions = scaler_labels.inverse_transform(mean_predictions.reshape(-1, 1)).squeeze()
            lower_bound = scaler_labels.inverse_transform(lower_bound.reshape(-1, 1)).squeeze()
            upper_bound = scaler_labels.inverse_transform(upper_bound.reshape(-1, 1)).squeeze()

        mae, rmse, mape, r2 = compute_metrics(all_labels, mean_predictions)
        print(f"评估指标 --> MAE -> {mae:.4f}\n"
              f"----------RMSE -> {rmse:.4f}\n"
              f"----------MAPE -> {mape:.4f}\n"
              f"------------R2 -> {r2:.4f}")
        return all_labels, mean_predictions, lower_bound, upper_bound
    else:
        mae, rmse, mape, r2 = compute_metrics(labels, predictions)
        print(f"评估指标 --> MAE -> {mae:.4f}\n"
              f"----------RMSE -> {rmse:.4f}\n"
              f"----------MAPE -> {mape:.4f}\n"
              f"------------R2 -> {r2:.4f}")
        return labels, predictions


def visualize_predictions(all_labels, all_predictions, save_path=None):
    """绘制预测图"""
    plt.figure(figsize=(14, 7))
    plt.plot(all_predictions, label="Pred", color='blue')
    plt.plot(all_labels, label="True Value", color='red', linestyle='dashed')
    plt.title("Predictions vs True Values")
    plt.xlabel("sample", fontsize=18)
    plt.ylabel("value", fontsize=18)
    plt.rcParams.update({'font.size': 16})
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    plt.show()


def visualize_predictions_interval(test_time, all_labels, all_predictions, lower_bound=None, upper_bound=None, save_path=None):
    # 定义窗口大小
    window_size = 12

    # 对真实值、预测值、上界和下界进行移动平均处理
    smoothed_labels = moving_average(all_labels, window_size)
    smoothed_predictions = moving_average(all_predictions, window_size)
    smoothed_lower_bound = moving_average(lower_bound, window_size)
    smoothed_upper_bound = moving_average(upper_bound, window_size)

    test_times = test_time[:len(smoothed_labels)]

    plt.figure(figsize=(12, 6))

    # Plot real values
    plt.plot(test_times, smoothed_labels, color='#000000', lw=2, label='Real Value')

    # Plot predicted values
    plt.plot(test_times, smoothed_predictions, color='#CF0A0A', linestyle='dashed', lw=2, label='Predicted Value')

    # Plot prediction interval
    plt.fill_between(test_times, smoothed_lower_bound, smoothed_upper_bound, color='#BCDAFC', alpha=0.5,
                     label='Prediction Interval')
    # 为间隔边框添加虚线
    plt.plot(test_times, smoothed_lower_bound, linestyle='dashed', linewidth=1.0, color='#0070C0')
    plt.plot(test_times, smoothed_upper_bound, linestyle='dashed', linewidth=1.0, color='#0070C0')

    # Set axis labels
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Carbon Price', fontsize=18)
    plt.rcParams.update({'font.size': 16})
    # 设置 x 轴和 y 轴刻度朝内
    plt.tick_params(axis='both', direction='in')

    # Add legend
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
