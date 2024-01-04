# GC-TCN

```python
"""
	两层GCN接全连接
"""
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

```

