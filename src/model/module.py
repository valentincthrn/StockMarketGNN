import torch

from torch_geometric.nn import GATConv
import torch.nn.functional as F


class CompanyExtractor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(CompanyExtractor, self).__init__()
        self.lstm = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=1
        ).to(device)

    def forward(self, x):
        _, hn = self.lstm(x)
        return hn[0]


class MLPWithHiddenLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, device):
        super(MLPWithHiddenLayer, self).__init__()

        self.hidden = torch.nn.Linear(input_size, input_size).to(device)
        self.output = torch.nn.Linear(input_size, output_size).to(device)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        return self.output(x)


class MyGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        device
    ):
        super(MyGNN, self).__init__()
        self.gat_conv = GATConv(in_channels=in_channels, out_channels=out_channels).to(device)
        self.device = device

    def forward(self, data):
        lstm_tensor = data
        nbr_nodes = lstm_tensor.shape[0]
        edge_index = torch.combinations(torch.arange(nbr_nodes)).t().to(self.device)

        x = self.gat_conv(lstm_tensor, edge_index)
        x = F.relu(x)

        return x
