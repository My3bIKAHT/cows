import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(7, 64)
        self.l2 = nn.Linear(64, 256)
        self.l3 = nn.Linear(256,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x
