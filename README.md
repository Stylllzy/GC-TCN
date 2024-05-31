# DGCPN

```python
"""
	两层GCN接全连接
"""
from torch_geometric.nn import GCNConv
from models.tcn import TemporalConvNet

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

```

