import torch
import torch.nn.functional as F


def psnr(prediction, target, max_value=1.0):
    mse = F.mse_loss(prediction, target)
    return 10.0 * torch.log10(max_value**2 / mse)


def ssim(prediction, target, window_size=11, sigma=1.5, max_value=1.0):
    prediction = prediction.reshape(-1, 1, 28, 28)
    target = target.reshape(-1, 1, 28, 28)

    coordinates = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gaussian = torch.exp(-(coordinates**2) / (2 * sigma**2))
    window = (gaussian[:, None] @ gaussian[None, :]).to(prediction.device)
    window = (window / window.sum()).expand(1, 1, window_size, window_size)

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    mu_prediction = F.conv2d(prediction, window)
    mu_target = F.conv2d(target, window)

    mu_prediction_sq = mu_prediction**2
    mu_target_sq = mu_target**2
    mu_product = mu_prediction * mu_target

    sigma_prediction_sq = F.conv2d(prediction**2, window) - mu_prediction_sq
    sigma_target_sq = F.conv2d(target**2, window) - mu_target_sq
    sigma_product = F.conv2d(prediction * target, window) - mu_product

    similarity = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (sigma_prediction_sq + sigma_target_sq + c2)
    )

    return similarity.mean()
