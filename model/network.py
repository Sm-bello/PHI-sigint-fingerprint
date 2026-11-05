"""
PHI-RF-CLASSIFIER: 1D Convolutional Neural Network (1D-CNN) Architecture
========================================================================
Processes 2-channel raw complex baseband I/Q tensors of shape (batch, 2, 1024)
through 3 hierarchical 1D convolutional blocks with BatchNorm, ReLU, MaxPooling,
and global AdaptiveAvgPool1d(1) before final 5-class linear classification.
"""
import torch
import torch.nn as nn


class RFClassifierCNN(nn.Module):
    """1D-CNN Deep Learning architecture for real-time battlefield RF emitter classification."""

    def __init__(self, num_classes: int = 5):
        super(RFClassifierCNN, self).__init__()
        # Input shape: (batch, 2, 1024) -> 2 channels (I and Q)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=16, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the 128-dimensional latent RF fingerprint embedding vector."""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.fc(features)
