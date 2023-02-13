import torch
import torch.nn as nn
from model.fast_rcnn_classifier import FastRCNNClassifier


class ImageEncoder(nn.Module):
    def __init__(self, FastRCNNClassifier):
        super(ImageEncoder, self).__init__()
        self.fast_rcnn = FastRCNNClassifier

    def forward(self, x):
        x = self.fast_rcnn(x)
        return x

