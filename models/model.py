import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from models.tcn import TemporalConvNet


class GCN_FC(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, num_classes):
        super(GCN_FC, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Fully connected layer
        self.fc = torch.nn.Linear(gcn_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, target_node_index = data.x, data.edge_index, data.target_node_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # [1088, gcn_hidden_dim]  [1088, 64]

        # 选择目标节点的特征进行预测
        target_node_feature = x[target_node_index]  # [32, gcn_hidden_dim]  [32, 64]

        # Fully connected layer
        out = self.fc(target_node_feature)  # [gcn_hidden_dim] -> [num_classes]

        return out


class GCN_LSTM(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, lstm_hidden_dim, num_classes, num_lstm_layers, dropout):
        super(GCN_LSTM, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # dropout
        self.dropout_rate = dropout

        # LSTM Layers
        self.lstm = torch.nn.LSTM(input_size=gcn_hidden_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers,
                                  batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, target_node_index = data.x, data.edge_index, data.target_node_index

        # GCN layers
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层

        # Select the target node feature
        target_node_feature = h[target_node_index]  # [32, gcn_hidden_dim]

        target_node_feature = target_node_feature.unsqueeze(1)  # [32, 1, gcn_hidden_dim]

        # LSTM layer
        lstm_out, _ = self.lstm(target_node_feature)

        # Fully connected layer
        out = self.fc(lstm_out.squeeze())  # Take the output of the last time step

        return out


class GCN_GRU(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, gru_hidden_dim, num_classes, num_gru_layers=1, dropout=0.5):
        super(GCN_GRU, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Dropout
        self.dropout_rate = dropout

        # GRU Layers
        self.gru = torch.nn.GRU(gcn_hidden_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True,
                                dropout=dropout if num_gru_layers > 1 else 0)

        # Fully connected layer
        self.fc = torch.nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, target_node_index = data.x, data.edge_index, data.target_node_index

        # GCN layers
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层

        # # 全局池化（将节点特征组合成图形表示）
        # x = global_mean_pool(x, data.batch)  # [num_graphs, gcn_hidden_dim]
        #
        # # LSTM 层需要输入形状 [batch_size、seq_len、input_size]，但由于我们的序列长度为 1（每个图），我们添加一个额外的维度
        # x = x.unsqueeze(0)  # [num_graphs, 1, gcn_hidden_dim]
        #
        # gru_out, _ = self.gru(x)
        #
        # # Fully connected layer
        # out = self.fc(gru_out.squeeze())  # [num_graphs, num_classes]

        # Select the target node feature
        target_node_feature = h[target_node_index]  # [32, gcn_hidden_dim]

        target_node_feature = target_node_feature.unsqueeze(1)  # [32, 1, gcn_hidden_dim]

        # LSTM layer
        lstm_out, _ = self.gru(target_node_feature)

        # Fully connected layer
        out = self.fc(lstm_out.squeeze())  # Take the output of the last time step

        return out


class GC_TCN(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, tcn_num_channels, num_classes, kernel_size=2, dropout=0.2,
                 dilation_size=None):
        super(GC_TCN, self).__init__()

        # GCN Layers
        if dilation_size is None:
            dilation_size = [1]
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Dropout
        self.dropout_rate = dropout

        # TCN Layer
        self.tcn = TemporalConvNet(num_inputs=gcn_hidden_dim, num_channels=tcn_num_channels, kernel_size=kernel_size,
                                   dropout=dropout, dilations=dilation_size)
        self.linear = nn.Linear(tcn_num_channels[-1], num_classes)  # tcn_num_channels[-1] 取列表最后一个通道数

    def forward(self, data):
        x, edge_index, target_node_index = data.x, data.edge_index, data.target_node_index
    # def forward(self, x, edge_index, target_node_index):
        # GCN layers
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  # dropout层

        # Global pooling
        # x = global_mean_pool(x, batch)  # Shape: [32, gcn_hidden_dim]
        #
        # # 准备 TCN 的序列（添加通道维度）
        # x = x.unsqueeze(2)  # Shape: [batch-size, gcn_hidden_dim, 1]
        #
        # # TCN layer
        # tcn_out = self.tcn(x)  # [32, 64, 1]
        # tcn_out = tcn_out[:, :, -1]  # Shape: [batch-size, gcn_hidden_dim]
        # out = self.linear(tcn_out)  # tcn_out [32, 64] -> out [32, 1]

        # 选择目标节点的特征进行预测
        target_node_feature = h[target_node_index]  # [32, gcn_hidden_dim]  [32, 64]

        # 准备 TCN 的序列（添加通道维度）
        target_node_features = target_node_feature.unsqueeze(2)  # Shape: [32, gcn_hidden_dim, 1]

        tcn_out = self.tcn(target_node_features)  # [32, 64, 1]
        # 使用最后一个时间步的输出
        tcn_out = tcn_out[:, :, -1]

        # Fully connected layer for classification or regression
        out = self.linear(tcn_out)

        return out


class GCN_LSTM_Probabilistic(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, lstm_hidden_dim, num_classes, num_lstm_layers=1, dropout=0.5):
        super(GCN_LSTM_Probabilistic, self).__init__()

        self.num_classes = num_classes

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # LSTM Layers
        self.lstm = torch.nn.LSTM(gcn_hidden_dim, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True,
                                  dropout=dropout if num_lstm_layers > 1 else 0)

        # Fully connected layers: One for mean and another for log variance
        self.fc_mean = torch.nn.Linear(lstm_hidden_dim, num_classes)
        self.fc_log_var = torch.nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, data.batch)

        # LSTM layer
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)

        # Fully connected layers for mean and log variance
        mean = self.fc_mean(lstm_out.squeeze(1))
        log_var = F.softplus(self.fc_log_var(lstm_out.squeeze(1)))  # softplus(x) = log(1 + exp(x))

        return mean, log_var
