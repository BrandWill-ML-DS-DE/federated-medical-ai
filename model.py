import torch
import torch.nn as nn
import torch.nn.functional as F

class MedicalNet(nn.Module):

    def __init__(self):
        super(MedicalNet, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)