import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from models import *


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


def read_image(img_file, cvt_code=cv.COLOR_BGR2RGB):
    """Read image and optionally convert color space.
    
    Parameters
    ----------
    img_file : str
        Path to image
    
    cvt_code : int, default=cv.COLOR_BGR2RGB (=4)
        Desired color space code
        
    Returns
    -------
    img :
        Image (color space conversion if specified)
    """
    img = cv.imread(img_file, cv.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv.cvtColor(img, cvt_code)
    return img


def show_image(
    img,
    box,
    box_fmt='ltwh',
    box_color=(0, 0, 255),
    box_thickness=3, 
    score_map=None,
    max_size=960, 
    window_title="", 
    delay=1, 
    cvt_code=cv.COLOR_RGB2BGR
    ):
    # Color conversion
    if cvt_code is not None:
        img = cv.cvtColor(img, cvt_code)

    # Resize image if too large
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale)
        )
        img = cv.resize(img, out_size)
        box = np.array(box, dtype=np.float32) * scale

    # Check box format
    box = np.array(box, dtype=np.int32)
    if box_fmt == 'ltrb':
        box[2:] -= box[:2]

    # Clip box if out of frame
    bound = np.array(img.shape[1::-1])
    box[:2] = np.clip(box[:2], 0, bound)
    box[2:] = np.clip(box[2:], 0, bound - box[:2])

    # Draw box
    top_left = (box[0], box[1])
    bot_right = (box[0] + box[2], box[1] + box[3])
    img = cv.rectangle(
        img,
        top_left,
        bot_right,
        box_color,
        box_thickness
    )
    
    # Concatenate score map (align with image height)
    if score_map is not None:
        scale = max_size / max(score_map.shape[1])
        out_size = (
            int(score_map.shape[1] * scale),
            int(score_map.shape[0] * scale)
        )
        score_map = cv.resize(score_map, out_size)
        
        # Reshape score_map
        
        # Convert color code
        
        # Horizontally concatenate to image
        img = np.concatenate((img, score_map), axis=1)
    
    # Display image
    cv.imshow("", img)
    cv.setWindowTitle("", window_title)
    cv.waitKey(delay)
    return img


def crop_and_resize(
    img, 
    center, 
    in_size, 
    out_size,
    border_type=cv.BORDER_CONSTANT,
    border_value=(0, 0, 0),
    interp=cv.INTER_LINEAR
    ):
    """Returns cropped and resized centered image.
    
    Parameters
    ----------
    img : ndarray of shape (H, W, 3)
        Original image
    
    center : ndarray of shape (2,)
        Center of bounding box (y, x)
        Coordinates with respect to top left of image
        
    in_size : int
        Size of input exemplar image
        
    out_size : int
        Size of ouput exemplar image
        
    border_type : default=cv.BORDER_CONSTANT
        Type of border when adding padding to image
    
    border_value : ndarray of shape (3,)
        Value to be used for border padding
    
    interp : default=cv.INTER_LINEAR
        Interpolation method to be used in resizing. 
        cv.INTER_CUBIC would provide higher resolution, but cv.INTER_LINEAR is faster.
    
    Returns
    -------
    patch : ndarray of shape (out_size, out_size)
        Cropped and resized image patch
    """
    # Calculate coordinates of corners of exemplar image in reference to original image
    in_size = np.round(in_size)
    top_left = np.round(center - (in_size - 1) / 2)
    corners = np.concatenate((top_left, top_left + in_size)).astype(int)

    # Corners of patch
    pads = np.concatenate((
        -corners[:2], 
        corners[2:] - img.shape[:2])
    )
    
    # Add padding (if necessary)
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv.copyMakeBorder(img, npad, npad, npad, npad, border_type, value=border_value)
        corners += npad

    # Crop image patch and resize
    patch = cv.resize(
        img[corners[0]:corners[2], corners[1]:corners[3]], 
        (out_size, out_size),
        interpolation=interp
    )
    return patch