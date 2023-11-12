"""
    配置方法模块
    常用替换参数
"""

# -------------------------------------- 数据参数 --------------------------------------
default_target = 'deal_sh'  # 预测目列
default_loader_path = './dataset/dataloader/d-sh-dataloader.pth'  # dataloader路径

# -------------------------------------- 模型参数 --------------------------------------
default_gcn_hidden_dim = 64  # GCN层隐藏层维度

default_lstm_hidden_dim = 64  # LSTM层隐藏层维度
default_num_lstm_layers = 2  # LSTM层数量

default_gru_hidden_dim = 16  # GRU层隐藏层维度
default_num_gru_layers = 2  # GRU层数量

default_tcn_num_channels = [16, 16, 16]  # TCN层通道数量
default_tcn_kernel_size = 3  # TCN层卷积核大小

# -------------------------------------- 训练参数 --------------------------------------
default_lr = 0.0001  # 学习率
default_num_epochs = 500  # 训练轮数
default_dropout = 0.3  # dropout rate
default_model = 'gcn-lstm'  # 模型名称
default_model_path = "./checkpoints/gcn-lstm/init/"  # 保存权重文件的路径
default_weight_name = 'd-sh-model.pth'  # 权重文件名
