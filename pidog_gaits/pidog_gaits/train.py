"""
Training Script for Gait Neural Network

Trains the neural network to predict joint angles from gait parameters.
Includes data loading, training loop, validation, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from datetime import datetime

from neural_network import GaitNet, GaitNetLarge, load_dataset, count_parameters


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', save_dir='./models'):
    """
    Train the gait model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cpu' or 'cuda'
        save_dir: Directory to save models

    Returns:
        dict: Training history
    """
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    print("\nStarting training...")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Model parameters: {count_parameters(model):,}")
    print("=" * 60)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f} "
                  f"Val Loss: {val_loss:.6f} "
                  f"LR: {history['lr'][-1]:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, best_model_path)

    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.6f}")

    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
    }, final_model_path)

    print(f"Saved best model to: {best_model_path}")
    print(f"Saved final model to: {final_model_path}")

    return history


def plot_training_history(history, save_path='training_history.png'):
    """Plot training and validation loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Learning rate plot
    ax2.plot(history['lr'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved training plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train PiDog Gait Neural Network')
    parser.add_argument('--data', type=str, required=True, help='Path to training data (.npz file)')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'large'],
                        help='Model architecture (simple or large)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--save_dir', type=str, default='./models', help='Model save directory')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, or cuda')

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("PiDog Gait Neural Network Training")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    dataset = load_dataset(args.data)

    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\nCreating model: {args.model}")
    if args.model == 'simple':
        model = GaitNet()
    else:
        model = GaitNetLarge()

    # Train model
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir
    )

    # Plot results
    plot_path = os.path.join(args.save_dir, 'training_history.png')
    plot_training_history(history, plot_path)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
