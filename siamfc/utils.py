import cv2
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
    
    # Enter labels (+1 if <= R, -1 otherwise)
    labels = np.where(
        dist <= r_pos, 
        np.ones_like(x), 
        np.zeros_like(x)
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

def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img

def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale
    
    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]
        
        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])
        
        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)
        
        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch