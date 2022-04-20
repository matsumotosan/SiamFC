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
    neg_mask = (labels == 0)
    
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

def triplet_loss(scores, labels):
    """Calculate triplet loss for predicted score map.
    
    Parameters
    ----------
    scores : torch.Tensor of shape (N, C, W, H)
        Predicted score map
        
    labels : torch.Tensor of shape (N, C, W, H)
        Ground truth score map
    
    Returns
    -------
    triplet loss: torch.Tensor
    """
    N = labels.shape[0]
    loss = 0

    for i in range(N):
        label = labels[i].flatten()
        score = scores[i].flatten()
        n = torch.sum(label==0).item() #Number of negative instances
        m = torch.sum(label==1).item() #Number of positive instances
        v_p = score[label==1].reshape(m, 1)  #Extract the positive score vector 
        v_n = score[label==0].reshape(n, 1)  #Extract the negative score vector
        V_p = torch.tile(v_p, (1, n))
        V_n = torch.tile(v_n, (1, m))
        V_p = V_p.T
        loss = loss+torch.sum(torch.log(1 + torch.exp(V_n - V_p))) / (m * n)

    return loss