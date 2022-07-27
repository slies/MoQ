#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


def AVG_classification_accuracy(true, pred, quantile=0.95):
    """
    This function is used to calculate the metrics "Accuracy" and "Sensitivity" reported in the paper.
    
    (Section: Benchmarks and Performance Metrics)
    
    """
    peak_val = np.quantile(true, quantile)
    peak_pos = np.where(true >= peak_val)
    non_peak_pos = np.where(true < peak_val)
    peak_pred = pred[peak_pos]
    non_peak_pred = pred[non_peak_pos]
    
    peak_detect = np.zeros_like(peak_pred)
    peak_detect[np.where(peak_pred >= peak_val)] = 1
    
    non_peak_detect = np.zeros_like(non_peak_pred)
    non_peak_detect[np.where(non_peak_pred < peak_val)] = 1
    
    accuracy_peak = np.sum(peak_detect) / peak_detect.size
    accuracy_non_peak = np.sum(non_peak_detect) / non_peak_detect.size
    accuracy_avg = (accuracy_peak + accuracy_non_peak)/2
    
    return accuracy_avg, (accuracy_non_peak, accuracy_peak)
    