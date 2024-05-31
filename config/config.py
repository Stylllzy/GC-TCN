"""
    配置方法模块
    常用替换参数
"""

# -------------------------------------- 数据参数 --------------------------------------
default_data_path = './dataset/tanjiaoyi/sh_carbon.csv'
default_target = 'close'  # 预测目列
default_loader_path = './dataset/tanjiaoyi/dataloader-64/test_sh.pth'  # dataloader路径

# -------------------------------------- 模型参数 --------------------------------------
default_gcn_hidden_dim = 128  # GCN层隐藏层维度

default_lstm_hidden_dim = 64  # LSTM层隐藏层维度
default_num_lstm_layers = 2  # LSTM层数量

default_gru_hidden_dim = 64  # GRU层隐藏层维度
default_num_gru_layers = 2  # GRU层数量

default_tcn_num_channels = [64,32,16]  # TCN层通道数量
default_tcn_kernel_size = 3  # TCN层卷积核大小
default_tcn_dilation_size = [1,8,16]  # TCN层膨胀系数 [1,1,1]、[1,1,2]、[1,2,4]、[1,4,8]、[1,8,16]

# -------------------------------------- 训练参数 --------------------------------------
default_lr = 0.005  # 学习率 todo
default_num_epochs = 200  # 训练轮数
default_dropout = 0.3  # dropout rate
default_model = 'gc-tcn'  # 模型名称
default_model_path = "./checkpoints/Hyperparameter_experiments/Figure-D/"  # 保存权重文件的路径
default_weight_name = (f'sh-gcn{default_gcn_hidden_dim}-lr{default_lr}-channel{default_tcn_num_channels}'
                       f'-d{default_tcn_dilation_size}.pth')  # 权重文件名 todo

default_savefig_path = './vis/修改-区间预测图/GT-hb.png'
