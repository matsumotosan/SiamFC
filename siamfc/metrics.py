import numpy as np


def calc_center_error(output, upscale_factor):
    """This metric measures the displacement between the estimated center of the target and the ground-truth

    Args:
        output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
        upscale_factor: (int) Indicates how much we must upscale the output feature map to match it to he input images

    Returns:
        c_error:(int) The center displacement in pixels
    """
    b = output.shape[0]
    s = output.shape[-1]
    out_flat = output.reshape(b, -1)
    max_idx = np.argmax(out_flat, axis=1)
    estim_center = np.stack([max_idx // s, max_idx % s], axis=1)
    dist = np.linalg.norm(estim_center -s // 2, axis=1)
    c_error = dist.mean()
    c_error = c_error * upscale_factor
    return c_error

def precision_auc():
    pass

def threshold(score_map, labels, threshold):
    """Determines if tracker is successful given a threshold"""
    pass

def mean_iou(score_map, labels):
    """Calculates mean IoU.

    https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

    Parameters
    ----------
    score_map : array of size (N, H, W)

    labels : array of size (N, H, W)

    Returns
    -------
    mean_iou : array of size (N,)
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