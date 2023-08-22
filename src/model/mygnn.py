import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.fc1 = torch.nn.Linear(hidden_channels, num_classes)  # MLP hidden layer

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.fc1(x)
        return x
