"""
PHI-RF-CLASSIFIER: Deep Learning 1D-CNN Package
Provides PyTorch dataset generation, 1D Convolutional Neural Network architecture, and training pipeline.
"""
from model.network import RFClassifierCNN
from model.dataset import RFIQDataset, build_splits

__all__ = ["RFClassifierCNN", "RFIQDataset", "build_splits"]
