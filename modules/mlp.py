# modules/mlp.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=512):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.rep_dim = output_dim

    def forward(self, x):
        return self.net(x)