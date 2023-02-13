import torch
import torch.nn as nn


class GaussKernelLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.biases = nn.Parameter(torch.randn(output_dim))

    def forward(self, inputs):
        # Apply Gaussian kernel
        output = torch.exp(-torch.sum((inputs.unsqueeze(1) - self.weights) ** 2, dim=2) / 2)
        # Add biases
        output = output + self.biases
        return output


class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, inputs):
        return self.linear(inputs)


class SpatialGraphClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, gauss_kernel_dim, embedding_dim):
        super().__init__()
        self.gauss_kernel = GaussKernelLayer(input_dim, gauss_kernel_dim)
        self.embedding = EmbeddingLayer(gauss_kernel_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, inputs):
        gauss_kernel_output = self.gauss_kernel(inputs)
        embedding_output = self.embedding(gauss_kernel_output)
        classifier_output = self.classifier(embedding_output)
        return classifier_output
