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

def bce_loss_balanced(scores, labels, neg_weight=1.0):
    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    
    pos_num = pos_mask.sum().float()
    neg_sum = neg_mask.sum().float()
    
    weight = labels.new_zeros(labels.size())
    weight[pos_mask] = 1 / pos_num
    weight[neg_mask] = 1 / neg_sum * neg_weight
    weight /= weight.sum()
    
    return F.binary_cross_entropy_with_logits(scores, labels, weight, reduction='mean')