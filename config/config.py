"""
    配置方法模块
    常用替换参数
"""

# -------------------------------------- 数据参数 --------------------------------------
default_target = 'close'  # 预测目列
default_loader_path = './dataset/tanjiaoyi/dataloader/add_policy_uncertainty/s-sh-dataloader.pth'  # dataloader路径

# -------------------------------------- 模型参数 --------------------------------------
default_gcn_hidden_dim = 112  # GCN层隐藏层维度

default_lstm_hidden_dim = 64  # LSTM层隐藏层维度
default_num_lstm_layers = 3  # LSTM层数量

default_gru_hidden_dim = 64  # GRU层隐藏层维度
default_num_gru_layers = 2  # GRU层数量

default_tcn_num_channels = [64, 32, 16]  # TCN层通道数量
default_tcn_kernel_size = 3  # TCN层卷积核大小

# -------------------------------------- 训练参数 --------------------------------------
default_lr = 0.0013114035403734698  # 学习率
default_num_epochs = 1000  # 训练轮数
default_dropout = 0.2  # dropout rate
default_model = 'gc-tcn'  # 模型名称
default_model_path = "./checkpoints/gc-tcn/tanjiaoyi/add_policy_uncertainty/"  # 保存权重文件的路径
default_weight_name = 'sh-model.pth'  # 权重文件名

default_savefig_path = './vis/GT-sh.png'
