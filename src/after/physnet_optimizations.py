"""PhysNet optimization implementations for before/after benchmarks.

This module provides equivalent functions/classes for the optimized variants
measured in the benchmark script.
"""

from functools import lru_cache

import numpy as np
import scipy
import torch
from scipy.sparse import spdiags
from torch import nn


class VectorizedNegPearson(nn.Module):
    """Batch-vectorized equivalent of PhysNet's negative Pearson loss."""

    def forward(self, preds, labels):
        preds = preds.view(preds.shape[0], -1)
        labels = labels.view(labels.shape[0], -1)

        sum_x = torch.sum(preds, dim=1)
        sum_y = torch.sum(labels, dim=1)
        sum_xy = torch.sum(preds * labels, dim=1)
        sum_x2 = torch.sum(preds.pow(2), dim=1)
        sum_y2 = torch.sum(labels.pow(2), dim=1)

        n = preds.shape[1]
        numerator = n * sum_xy - sum_x * sum_y
        denominator = torch.sqrt(
            (n * sum_x2 - sum_x.pow(2)) * (n * sum_y2 - sum_y.pow(2))
        )
        eps = torch.finfo(preds.dtype).eps
        pearson = numerator / denominator.clamp_min(eps)
        return torch.mean(1 - pearson)


def original_neg_pearson_reference(preds, labels):
    """Reference implementation matching the original PhysNet loop."""
    loss = 0
    for i in range(preds.shape[0]):
        sum_x = torch.sum(preds[i])
        sum_y = torch.sum(labels[i])
        sum_xy = torch.sum(preds[i] * labels[i])
        sum_x2 = torch.sum(torch.pow(preds[i], 2))
        sum_y2 = torch.sum(torch.pow(labels[i], 2))
        n = preds.shape[1]
        pearson = (n * sum_xy - sum_x * sum_y) / (
            torch.sqrt((n * sum_x2 - torch.pow(sum_x, 2)) * (n * sum_y2 - torch.pow(sum_y, 2)))
        )
        loss += 1 - pearson
    return loss / preds.shape[0]


@lru_cache(maxsize=32)
def _detrend_projection(signal_length, lambda_value):
    """Cache the dense projection matrix used by the original detrend helper."""
    signal_length = int(signal_length)
    lambda_value = float(lambda_value)
    h = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d_mat = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    return h - np.linalg.inv(h + (lambda_value**2) * np.dot(d_mat.T, d_mat))


def cached_detrend(input_signal, lambda_value):
    """Detrend with the same formula as rPPG-Toolbox, reusing length-specific matrices."""
    input_signal = np.asarray(input_signal)
    projection = _detrend_projection(input_signal.shape[0], float(lambda_value))
    return np.dot(projection, input_signal)


def original_detrend_reference(input_signal, lambda_value):
    """Reference implementation matching rPPG-Toolbox evaluation.post_process._detrend."""
    input_signal = np.asarray(input_signal)
    signal_length = input_signal.shape[0]
    h = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d_mat = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (h - np.linalg.inv(h + (lambda_value**2) * np.dot(d_mat.T, d_mat))), input_signal
    )
    return detrended_signal


def fft_circular_macc(pred_signal, gt_signal):
    """Compute circular MACC over lags 0..N-2 using FFT-based correlations.

    The original implementation computes abs(corrcoef(pred, roll(gt, lag))) for
    every lag in range(0, len(pred)-1). This preserves that lag range while
    reducing repeated Python-level corrcoef calls.
    """
    pred = np.asarray(pred_signal, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt_signal, dtype=np.float64).reshape(-1)
    min_len = min(pred.size, gt.size)
    pred = pred[:min_len]
    gt = gt[:min_len]
    if min_len < 2:
        return float("nan")

    pred_centered = pred - np.mean(pred)
    gt_centered = gt - np.mean(gt)
    pred_norm = np.linalg.norm(pred_centered)
    gt_norm = np.linalg.norm(gt_centered)
    if pred_norm == 0.0 or gt_norm == 0.0:
        return float("nan")

    corr = scipy.fft.ifft(scipy.fft.fft(pred_centered) * np.conj(scipy.fft.fft(gt_centered))).real
    corr = np.abs(corr / (pred_norm * gt_norm))
    return float(np.max(corr[: min_len - 1]))


def original_macc_reference(pred_signal, gt_signal):
    """Reference implementation matching rPPG-Toolbox _compute_macc."""
    pred = np.asarray(pred_signal).reshape(-1)
    gt = np.asarray(gt_signal).reshape(-1)
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred) - 1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    return max(tlcc_list)
