import torch

from torch_geometric.nn import GATConv
import torch.nn.functional as F


class CompanyExtractor(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CompanyExtractor, self).__init__()
        self.lstm = torch.nn.RNN(
            input_size=input_size, hidden_size=hidden_size, num_layers=1
        )

    def forward(self, x):
        _, hn = self.lstm(x)
        return hn[0]


class MyGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(MyGNN, self).__init__()
        self.gat_conv = GATConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, data):
        lstm_tensor = data
        nbr_nodes = lstm_tensor.shape[0]
        edge_index = torch.combinations(torch.arange(nbr_nodes)).t()

        x = self.gat_conv(lstm_tensor, edge_index)
        x = F.relu(x)

        return x
