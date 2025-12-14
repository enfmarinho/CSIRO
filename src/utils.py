import matplotlib.pyplot as plt
import numpy as np
import torch

def plot(output_path, train_losses, val_losses):
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.plot(train_losses, label='Train Loss', marker='o', markersize=3)
        plt.plot(val_losses, label='Val Loss', marker='s', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
