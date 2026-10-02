import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class HerbDrugHGNN(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # First Graph Convolution Layer
        self.conv1 = GCNConv(5, 16)

        # Second Graph Convolution Layer
        self.conv2 = GCNConv(16, 8)

    def forward(self, x, edge_index):

        # Layer 1
        x = self.conv1(x, edge_index)

        x = F.relu(x)

        # Layer 2
        x = self.conv2(x, edge_index)

        return x