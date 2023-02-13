import torch
import torch.nn as nn
import torch.nn.functional as F


class FastRCNNClassifier(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_classes):
        super(FastRCNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.fc = nn.Linear(hidden_channels * 7 * 7, output_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
