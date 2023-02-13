import torch
import torch.nn as nn


class ExplanationGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ExplanationGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.embedding_layer = nn.Linear(input_size, hidden_size)
        self.cross_attention = nn.Linear(hidden_size, hidden_size)
        self.feed_forward = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Pass the input through the embedding layer
        x = self.embedding_layer(x)

        # Pass the input through multiple layers of cross-attention and feed-forward
        for _ in range(self.num_layers):
            attention = self.cross_attention(x)
            x = x + attention
            x = torch.relu(x)

            feed_forward = self.feed_forward(x)
            x = x + feed_forward
            x = torch.relu(x)

        # Pass the final result through the output layer to produce the text explanation
        x = self.output_layer(x)
        return x
