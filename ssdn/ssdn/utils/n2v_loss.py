from torch import Tensor

def loss_mse(masked_coords: list, true: torch.Tensor, pred: Tensor):
    mse = 0
    for coord in masked_coords:
        x, y = coord
        mse += ((true[:, :, x, y] - pred[:, :, x, y]) ** 2).mean().item()
    return mse / len(masked_coords)