import torch
import torch.nn as nn
import torch.nn.functional as F


def loss(score_map, labels):
    """Calculates binary cross-entropy loss for a score map."""
    
    return F.binary_cross_entropy()