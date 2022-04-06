import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss_balanced(scores, labels):
    """Calculate binary cross-entropy loss for predicted score map.
    
    Parameters
    ----------
    scores : torch.Tensor of shape (N, C, W, H)
        Predicted score map
        
    labels : torch.Tensor of shape (N, C, W, H)
        Ground truth score map
    
    Returns
    -------
    bce_loss : torch.Tensor 
        Binary cross-entropy loss
    """
    # Calculate mask of positive and negative examples
    pos_mask = (labels == 1)
    neg_mask = (labels == -1)
    
    # Calculate number of positive and negative examples
    pos_num = pos_mask.sum().float()
    neg_num = neg_mask.sum().float()
    
    # Calclulate rescaling weight for each element (scale to sum weights to one)
    weight = labels.new_zeros(labels.size())
    weight[pos_mask] = 1 / pos_num
    weight[neg_mask] = 1 / neg_num
    weight /= weight.sum()
    
    # Calculate BCE loss
    loss = F.binary_cross_entropy_with_logits(scores, labels, weight, reduction='mean')
    
    return loss