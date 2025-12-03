"""
Metrics and calibration utilities for the experiment.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


def calculate_ece(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.
    
    y_true: 1 if prediction was correct, 0 otherwise.
    y_prob: model confidence in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    
    if len(y_true_arr) == 0:
        return 0.0

    assert y_true_arr.shape == y_prob_arr.shape

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns indices 1..n_bins, we want 0..n_bins-1
    bin_ids = np.digitize(y_prob_arr, bins) - 1
    
    # Handle edge case where prob is exactly 1.0 (digitize puts it in next bin)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    total = len(y_prob_arr)

    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue

        bin_true = y_true_arr[mask]
        bin_prob = y_prob_arr[mask]

        avg_conf = float(bin_prob.mean())
        avg_acc = float(bin_true.mean())
        
        weight = float(mask.sum()) / float(total)
        ece += weight * abs(avg_acc - avg_conf)

    return float(ece)


def calibration_bins(
    y_true: List[int],
    y_prob: List[float],
    n_bins: int = 10
) -> List[Dict[str, float]]:
    """
    Compute bin-level statistics for reliability diagrams.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    
    if len(y_true_arr) == 0:
        return []

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    stats = []
    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        
        if count > 0:
            bin_true = y_true_arr[mask]
            bin_prob = y_prob_arr[mask]
            avg_conf = float(bin_prob.mean())
            avg_acc = float(bin_true.mean())
        else:
            avg_conf = 0.0
            avg_acc = 0.0

        stats.append({
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i+1]),
            "avg_conf": avg_conf,
            "avg_acc": avg_acc,
            "count": count
        })
        
    return stats



