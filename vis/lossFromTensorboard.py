# -*- coding: utf-8 -*-

from tensorboard.backend.event_processing import event_accumulator
from utils.util import plot_training_metrics

# 路径到你的TensorBoard日志文件
log_path = '../runs/hb64-Feb10_16-37-09_autodl-container-1be311953c-a987e89b'

# 加载事件文件
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()  # 加载所有数据

# 提取损失和R^2值
train_loss_values = ea.scalars.Items('Loss/Train')
val_loss_values = ea.scalars.Items('Loss/Val')
train_r2_values = ea.scalars.Items('R2/Train')
val_r2_values = ea.scalars.Items('R2/Val')

# 提取数据到列表
epochs = [x.step for x in train_loss_values]
train_loss = [x.value for x in train_loss_values]
val_loss = [x.value for x in val_loss_values]
train_r2 = [x.value for x in train_r2_values]
val_r2 = [x.value for x in val_r2_values]

# 绘制图表
plot_training_metrics(epochs, train_loss, val_loss, train_r2, val_r2)