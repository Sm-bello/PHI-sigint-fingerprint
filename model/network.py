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
    """1D-CNN Deep Learning architecture for real-time battlefield RF emitter classification.

    Operates directly on time-domain complex baseband In-Phase (I) and Quadrature (Q)
    samples arranged as a 2-channel 1D tensor of shape `(batch_size, 2, 1024)`.
    Extracts hierarchical cyclostationary phase, instantaneous frequency, and
    modulation envelope features across three convolutional stages before global
    average pooling and linear classification.

    Attributes:
        conv1 (nn.Conv1d): First 1D convolutional layer (2 -> 32 channels, kernel_size=16, stride=2).
        bn1 (nn.BatchNorm1d): Batch normalization for the 32-channel feature map.
        conv2 (nn.Conv1d): Second 1D convolutional layer (32 -> 64 channels, kernel_size=8, stride=2).
        bn2 (nn.BatchNorm1d): Batch normalization for the 64-channel feature map.
        conv3 (nn.Conv1d): Third 1D convolutional layer (64 -> 128 channels, kernel_size=4, stride=2).
        bn3 (nn.BatchNorm1d): Batch normalization for the 128-channel feature map.
        pool (nn.MaxPool1d): Temporal max-pooling layer (kernel_size=2, stride=2).
        relu (nn.ReLU): Rectified Linear Unit activation function.
        global_pool (nn.AdaptiveAvgPool1d): Shift-invariant global average pooling to length 1.
        fc (nn.Linear): Final classification head mapping the 128-D embedding to `num_classes`.
    """

    def __init__(self, num_classes: int = 5) -> None:
        """Initializes the RFClassifierCNN architecture.

        Args:
            num_classes (int, optional): Number of target RF emitter classes. Defaults to 5
                (`Background_Noise`, `Friendly_Comms`, `Hostile_MAVLink`, `Hostile_OcuSync`, `Jammer`).
        """
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
        """Extracts the 128-dimensional latent RF fingerprint embedding vector.

        Args:
            x (torch.Tensor): Input baseband I/Q tensor of shape `(batch_size, 2, 1024)`.

        Returns:
            torch.Tensor: Shift-invariant latent embedding tensor of shape `(batch_size, 128)`.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes unnormalized class logits for a batch of 2-channel I/Q waveforms.

        Args:
            x (torch.Tensor): Input baseband I/Q tensor of shape `(batch_size, 2, 1024)`.

        Returns:
            torch.Tensor: Output logit tensor of shape `(batch_size, num_classes)`.
        """
        features = self.extract_features(x)
        return self.fc(features)
