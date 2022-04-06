import numpy as np
import torch
import torch.nn.functional as F


def create_labels(size, k, r_pos, r_neg=0, metric='l1'):
    """Create ground truth score map.
    
    Originally formulated as:
        y[u] = +1 if k * ||u - c || <= R, otherwise -1.
    
    In implementation, we choose to create the labels as:
        y[u] = +1 if ||u - c || <= (R / k), otherwise -1.
    
    where u=(x,y) is the position of element on the score map.
    
    Parameters
    ----------
    size : ndarray of shape (N, C, H, W)
        Size of score map
    
    k : int
        Stride of the encoder network
    
    r_pos : int
        Radius threshold to be considered a positive example
    
    r_neg : int
    
    metric : str, default='l1'
        Metric for calculating distance to center, {'l1', 'l2'}
    
    Returns
    -------
    labels : ndarray of shape (N, C, H, W)
        Ground truth score map
    """
    # Create meshgrid of response map coordinates
    n, c, h, w = size
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    xx, yy = np.meshgrid(x, y)
    xy = np.stack((xx, yy), axis=0)

    # Calculate scaled distance threshold (R / k)
    r_pos /= k
    r_neg /= k
    
    # Calculate distance to center
    if metric == "l1":
        dist = np.linalg.norm(xy, ord=1, axis=0)
    elif metric == 'l2':
        dist = np.linalg.norm(xy, ord='fro', axis=0)
    else:
        return ValueError("Invalid distance metric.")
    
    # Determine labels based on distance
    # labels = np.where(
    #     dist <= r_pos, 
    #     np.ones_like(x), 
    #     np.where(
    #         dist < r_neg, 
    #         np.ones_like(x) * 0.5,
    #         np.zeros_like(x)
    #     )
    # )
    labels = np.where(
        dist <= r_pos, 
        np.ones_like(x), 
        -np.ones_like(x)
    )

    # Reshape to match size of score map
    labels = labels.reshape((1, 1, h, w))
    labels = np.tile(labels, (n, c, 1, 1))
    
    return labels

def xcorr(z, x, scale_factor=None, mode='bicubic'):
    """Calculates cross-correlation between exemplar exemplar and search image embeddings.
    
    Parameters
    ----------
    z : ndarray of shape (N, C, Hz, Wz)
        Exemplar images embeddings
    
    x : ndarray of shape (N, C, Hx, Wx)
        Search images embeddings
    
    scale_factor: int, default=None
        Upsampling scaling factor (same in all spatial dimensions)
        Bertinetto et al. set to 16 during tracking (17, 17) -> (272, 272)
        Can be set to 'None' (implicitly equal to 1) during training
    
    mode : str, default='bicubic'
        Upsampling interpolation method
        Choose from {'linear', 'bilinear', 'bicubic', 'trilinear', False}.
    
    Returns
    -------
    score_map : ndarray of shape (N, 1, Hmap * scale_factor, Wmap * scale_factor)
        Score map
        
    References
    ----------
    https://pytorch.org/docs/stable/generated/torch.nn.functional.upsample.html#torch.nn.functional.upsample
    """
    # Get tensor dimensions of exemplar and search embeddings
    nz = z.shape[0]
    nx, cx, hx, wx, = x.shape
    # assert nz == nx, "Minibatch sizes not equal."
    
    # Calculate cross-correlation
    x = x.view(-1, nz * cx, hx, wx)
    score_map = F.conv2d(x, z, groups=nz)
    score_map = score_map.view(nx, -1, score_map.shape[-2], score_map.shape[-1])
    
    # Upsample response map (N, C, H, W) -> (N, C, H * scale_factor, W * scale_factor)
    if scale_factor is not None:
        score_map = F.upsample(score_map, scale_factor=scale_factor, mode=mode)
    
    return score_map