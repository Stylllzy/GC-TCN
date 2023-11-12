import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv, global_mean_pool
from models.tcn import TemporalConvNet


class GCN_LSTM(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, lstm_hidden_dim, num_classes, num_lstm_layers=1, dropout=0.5):
        super(GCN_LSTM, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Dropout
        self.dropout = dropout

        # LSTM Layers
        self.lstm = torch.nn.LSTM(gcn_hidden_dim, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True,
                                  dropout=dropout if num_lstm_layers > 1 else 0)

        # Fully connected layer
        self.fc = torch.nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # [1248, gcn_hidden_dim]
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局池化（将节点特征组合成图形表示）
        x = global_mean_pool(x, data.batch)  # [32, gcn_hidden_dim]

        # LSTM 层需要输入形状 [batch_size、seq_len、input_size]，但由于我们的序列长度为 1（每个图），我们添加一个额外的维度
        x = x.unsqueeze(1)  # [32, 1, gcn_hidden_dim]

        lstm_out, _ = self.lstm(x)
        # lstm_out [32, 1, 64]

        # Fully connected layer
        out = self.fc(lstm_out.squeeze(1))  # [32, 64] -> [32, 1]

        return out


class GCN_GRU(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, gru_hidden_dim, num_classes, num_gru_layers=1, dropout=0.5):
        super(GCN_GRU, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Dropout
        self.dropout = dropout

        # GRU Layers
        self.gru = torch.nn.GRU(gcn_hidden_dim, gru_hidden_dim, num_layers=num_gru_layers, batch_first=True,
                                dropout=dropout if num_gru_layers > 1 else 0)

        # Fully connected layer
        self.fc = torch.nn.Linear(gru_hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局池化（将节点特征组合成图形表示）
        x = global_mean_pool(x, data.batch)  # [num_graphs, gcn_hidden_dim]

        # LSTM 层需要输入形状 [batch_size、seq_len、input_size]，但由于我们的序列长度为 1（每个图），我们添加一个额外的维度
        x = x.unsqueeze(1)  # [num_graphs, 1, gcn_hidden_dim]

        gru_out, _ = self.gru(x)

        # Fully connected layer
        out = self.fc(gru_out.squeeze(1))  # [num_graphs, num_classes]

        return out


class GC_TCN(torch.nn.Module):
    def __init__(self, num_node_features, gcn_hidden_dim, tcn_num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(GC_TCN, self).__init__()

        # GCN Layers
        self.conv1 = GCNConv(num_node_features, gcn_hidden_dim)
        self.conv2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # Dropout
        self.dropout = dropout

        # TCN Layer
        self.tcn = TemporalConvNet(gcn_hidden_dim, tcn_num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(tcn_num_channels[-1], num_classes)  # tcn_num_channels[-1] 取列表最后一个通道数

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # [1248, gcn_hidden_dim]
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, data.batch)  # Shape: [32, gcn_hidden_dim]

        # 准备 TCN 的序列（添加通道维度）
        x = x.unsqueeze(2)  # Shape: [batch-size, gcn_hidden_dim, 1]

        # TCN layer
        tcn_out = self.tcn(x)  # [32, 64, 1]
        tcn_out = tcn_out[:, :, -1]  # Shape: [batch-size, gcn_hidden_dim]
        out = self.linear(tcn_out)  # tcn_out [32, 64] -> out [32, 1]

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
