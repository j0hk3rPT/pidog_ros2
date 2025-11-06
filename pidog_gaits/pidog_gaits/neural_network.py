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


class GaitNetLSTM(nn.Module):
    """
    LSTM-based network for gait generation with temporal memory.

    Better for sim-to-real transfer as it can handle:
    - Servo lag (15-30ms delay)
    - Temporal dependencies in gaits
    - Velocity information

    Input: [gait_cmd (4), joint_pos (8), joint_vel (8)] = 20 features
    LSTM: 128 hidden units
    Output: 8 joint angles
    """

    def __init__(self, input_size=20, hidden_size=128, output_size=8, num_layers=1):
        """
        Initialize LSTM network.

        Args:
            input_size (int): Size of input features (4 + 8 + 8 = 20 by default)
                [gait_type, direction, turn, phase, 8 joint positions, 8 joint velocities]
            hidden_size (int): Number of LSTM hidden units
            output_size (int): Number of outputs (8 joint angles)
            num_layers (int): Number of LSTM layers
        """
        super(GaitNetLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Fully connected layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
               or (batch_size, input_size) for single timestep
            hidden: Optional hidden state (h, c) from previous timestep

        Returns:
            output: Predictions of shape (batch_size, output_size)
            hidden: New hidden state (h, c)
        """
        # Handle single timestep input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state with zeros."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return (h, c)


class GaitNetSimpleLSTM(nn.Module):
    """
    Simpler LSTM network that only uses gait commands (no state feedback).

    Good for initial training before adding full state feedback.
    Input: [gait_type, direction, turn, phase] = 4 features
    LSTM: 64 hidden units
    Output: 8 joint angles
    """

    def __init__(self, input_size=4, hidden_size=64, output_size=8):
        super(GaitNetSimpleLSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x, hidden=None):
        """Forward pass."""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out[:, -1, :])

        return output, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """Initialize hidden state."""
        h = torch.zeros(1, batch_size, self.hidden_size).to(device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(device)
        return (h, c)


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

    print("\n" + "=" * 60)
    print("GaitNetSimpleLSTM (For Sim-to-Real)")
    print("=" * 60)
    model_lstm_simple = GaitNetSimpleLSTM()
    print(model_lstm_simple)
    print(f"\nTotal parameters: {count_parameters(model_lstm_simple):,}")

    # Test LSTM forward pass
    dummy_input_lstm = torch.randn(10, 4)
    output_lstm, hidden = model_lstm_simple(dummy_input_lstm)
    print(f"\nInput shape: {dummy_input_lstm.shape}")
    print(f"Output shape: {output_lstm.shape}")

    print("\n" + "=" * 60)
    print("GaitNetLSTM (Full State Feedback)")
    print("=" * 60)
    model_lstm = GaitNetLSTM(input_size=20)  # 4 + 8 + 8
    print(model_lstm)
    print(f"\nTotal parameters: {count_parameters(model_lstm):,}")

    # Test with full state
    dummy_input_full = torch.randn(10, 20)  # gait(4) + pos(8) + vel(8)
    output_full, hidden_full = model_lstm(dummy_input_full)
    print(f"\nInput shape: {dummy_input_full.shape}")
    print(f"Output shape: {output_full.shape}")
