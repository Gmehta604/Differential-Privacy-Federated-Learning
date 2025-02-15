import torch

def add_noise(model, epsilon):
    """Add Gaussian noise to model parameters for differential privacy."""
    sensitivity = 1.0
    noise_scale = sensitivity / epsilon
    
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * noise_scale
            param.add_(noise)