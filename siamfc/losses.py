import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(scores, labels, weights):
    """Calculates binary cross-entropy loss for a score map.
    
    Parameters
    ----------
    scores : torch.Tensor of shape ()
        Predicted score map
        
    labels : torch.Tensor of shape ()
        Label
        
    weights : torch.Tensor of shape ()
        Weights
        
    Returns
    -------
    loss : float
        Binary cross entropy loss with logits for predicted score map and labels
    """
    return F.binary_cross_entropy_with_logits(scores, labels, weights, reduction='mean')