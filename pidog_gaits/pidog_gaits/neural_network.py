"""
Neural Network Model for Gait Generation

Simple feedforward network that learns to generate joint angles
from gait parameters (type, direction, turn, phase).

Architecture:
    Input: [gait_type, direction, turn, phase] (4 features)
    Hidden: Multiple fully-connected layers with ReLU activation
    Output: [8 joint angles] in radians
"""

import torch
import torch.nn as nn


class GaitNet(nn.Module):
    """
    Neural network for gait generation.

    This is a simple Multi-Layer Perceptron (MLP) that learns
    the mapping: gait_params -> joint_angles
    """

    def __init__(self, input_size=4, output_size=8, hidden_sizes=[128, 256, 128]):
        """
        Initialize the network.

        Args:
            input_size (int): Number of input features (default: 4)
                [gait_type, direction, turn, phase]
            output_size (int): Number of outputs (default: 8)
                [8 joint angles]
            hidden_sizes (list): Sizes of hidden layers
        """
        super(GaitNet, self).__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        return self.network(x)


class GaitNetLarge(nn.Module):
    """
    Larger version of GaitNet with more capacity.

    Use this if the simple model doesn't learn well enough.
    """

    def __init__(self, input_size=4, output_size=8):
        super(GaitNetLarge, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),

            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.network(x)


class GaitDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for gait training data.
    """

    def __init__(self, inputs, outputs):
        """
        Initialize dataset.

        Args:
            inputs: NumPy array of shape (N, 4) - input features
            outputs: NumPy array of shape (N, 8) - target joint angles
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def load_dataset(data_path):
    """
    Load dataset from .npz file.

    Args:
        data_path (str): Path to .npz file

    Returns:
        GaitDataset instance
    """
    import numpy as np

    data = np.load(data_path)
    inputs = data['inputs']
    outputs = data['outputs']

    print(f"Loaded dataset from {data_path}")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Outputs shape: {outputs.shape}")

    return GaitDataset(inputs, outputs)


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("=" * 60)
    print("GaitNet (Simple)")
    print("=" * 60)
    model = GaitNet()
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy_input = torch.randn(10, 4)  # Batch of 10
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("GaitNetLarge")
    print("=" * 60)
    model_large = GaitNetLarge()
    print(model_large)
    print(f"\nTotal parameters: {count_parameters(model_large):,}")
