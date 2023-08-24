import torch


def mape_loss(x, y):
    # Ensure no division by zero
    epsilon = 1e-6
    loss = torch.abs((x - y) / (y + epsilon))
    return torch.mean(loss)


def calculate_mape(true_values, pred_values):
    # Avoid division by zero
    mask = true_values != 0
    return 100 * (abs((true_values - pred_values) / true_values)[mask].mean())
