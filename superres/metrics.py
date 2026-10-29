"""Documented luminance PSNR and Gaussian-window SSIM with scale-border shaving."""

import math

import torch
from torch.nn import functional as F


def quantize(x):
    return x.clamp(0, 1).mul(255).round().div(255)


def luminance(x):
    # BT.601 limited-range Y, normalized to 0:1, matching common MATLAB rgb2ycbcr conventions.
    coeff = x.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1) / 255
    return (x * coeff).sum(1, keepdim=True) + 16 / 255


def psnr_y(sr, hr, scale):
    if sr.shape != hr.shape:
        raise ValueError("Matching shapes required")
    a, b = luminance(quantize(sr.float())), luminance(quantize(hr.float()))
    if min(a.shape[-2:]) <= 2 * scale:
        raise ValueError("Image too small for border shave")
    if scale:
        a, b = a[..., scale:-scale, scale:-scale], b[..., scale:-scale, scale:-scale]
    mse = (a - b).square().mean().item()
    return float("inf") if mse == 0 else -10 * math.log10(mse)


def psnr_ssim(sr, hr, scale, y_channel=True):
    if sr.shape != hr.shape or sr.ndim != 4:
        raise ValueError("Matching BCHW images required")
    a, b = quantize(sr.float()), quantize(hr.float())
    if y_channel:
        a, b = luminance(a), luminance(b)
    if scale:
        a, b = a[..., scale:-scale, scale:-scale], b[..., scale:-scale, scale:-scale]
    if min(a.shape[-2:]) < 11:
        raise ValueError("Image too small for 11x11 SSIM after border shave")
    mse = (a - b).square().mean().item()
    # Infinity is represented as null in JSON by the caller, with an explicit exact-match flag.
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    t = torch.arange(11, device=a.device, dtype=a.dtype) - 5
    g = torch.exp(-t.square() / (2 * 1.5**2))
    g /= g.sum()
    kernel = (g[:, None] * g[None, :]).expand(a.shape[1], 1, 11, 11)

    def filt(x):
        return F.conv2d(x, kernel, groups=a.shape[1])

    mu_a, mu_b = filt(a), filt(b)
    va, vb = filt(a.square()) - mu_a.square(), filt(b.square()) - mu_b.square()
    cov = filt(a * b) - mu_a * mu_b
    ssim = (
        (
            (2 * mu_a * mu_b + 0.01**2)
            * (2 * cov + 0.03**2)
            / ((mu_a.square() + mu_b.square() + 0.01**2) * (va + vb + 0.03**2))
        )
        .mean()
        .item()
    )
    return psnr, ssim
