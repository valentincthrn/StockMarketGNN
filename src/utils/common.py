import torch


def mape_loss(x, y):
    # Ensure no division by zero
    epsilon = 1e-6
    loss = torch.abs((x - y) / (y + epsilon))
    return torch.mean(loss)
