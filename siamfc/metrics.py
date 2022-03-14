import numpy as np


def precision_auc():
    pass

def threshold(score_map, labels, threshold):
    """Determines if tracker is successful given a threshold"""
    pass

def mean_iou(score_map, labels):
    """Calculates mean IoU.
    
    Parameters
    ----------
    score_map : array of size (B, H, W)
    
    labels : array of size (B, H, W)
    
    Returns
    -------
    mean_iou : array of size (B,)
    """
    intersection = (score_map & labels).sum((1, 2))
    union = (score_map | labels).sum((1, 2))
    mean_iou = (intersection / union).mean()
    return mean_iou

def accuracy(score_map, label):
    pass

def overlap():
    pass

def n_failures():
    pass