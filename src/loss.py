import torch.nn as nn
import torch

class WeightedMSELoss(nn.Module):
    """
    Calculates Mean Squared Error (MSE) with target-specific weights.
    
    The standard competition loss usually heavily weights Dry_Total_g and GDM_g.
    """
    def __init__(self, target_weights, device):
        super().__init__()
        self.weights = torch.tensor(target_weights, dtype=torch.float32).to(device)

    def forward(self, inputs, targets):
        # Calculate the squared error
        squared_error = (inputs - targets) ** 2
        
        # Apply the element-wise weights
        weighted_squared_error = squared_error * self.weights
        
        # Calculate the mean across the batch and targets
        loss = torch.mean(weighted_squared_error)
        
        return loss
