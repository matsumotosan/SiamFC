import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image
from models import *


fig_dict = {}
patch_dict = {}


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
    img = cv.imread(img_file, cv.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv.cvtColor(img, cvt_code)
    
    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv.resize(img, out_size)
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
            img = cv.rectangle(img, pt1, pt2, color.tolist(), thickness)
    
    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv.imshow(winname, img)
        cv.waitKey(delay)

    return img


def show_frame(image, boxes=None, fig_n=1, pause=0.001,
               linewidth=3, cmap=None, colors=None, legends=None) -> None:
    """Visualize an image w/o drawing rectangle(s).
    
    Args:
        image (numpy.ndarray or PIL.Image): Image to show.
        boxes (numpy.array or a list of numpy.ndarray, optional): A 4 dimensional array
            specifying rectangle [left, top, width, height] to draw, or a list of arrays
            representing multiple rectangles. Default is ``None``.
        fig_n (integer, optional): Figure ID. Default is 1.
        pause (float, optional): Time delay for the plot. Default is 0.001 second.
        linewidth (int, optional): Thickness for drawing the rectangle. Default is 3 pixels.
        cmap (string): Color map. Default is None.
        color (tuple): Color of drawed rectanlge. Default is None.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., ::-1])

    if not fig_n in fig_dict or \
        fig_dict[fig_n].get_size() != image.size[::-1]:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    if boxes is not None:
        if not isinstance(boxes, (list, tuple)):
            boxes = [boxes]
        
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y'] + \
                list(mcolors.CSS4_COLORS.keys())
        elif isinstance(colors, str):
            colors = [colors]

        if not fig_n in patch_dict:
            patch_dict[fig_n] = []
            for i, box in enumerate(boxes):
                patch_dict[fig_n].append(patches.Rectangle(
                    (box[0], box[1]), box[2], box[3], linewidth=linewidth,
                    edgecolor=colors[i % len(colors)], facecolor='none',
                    alpha=0.7 if len(boxes) > 1 else 1.0))
            for patch in patch_dict[fig_n]:
                fig_dict[fig_n].axes.add_patch(patch)
        else:
            for patch, box in zip(patch_dict[fig_n], boxes):
                patch.set_xy((box[0], box[1]))
                patch.set_width(box[2])
                patch.set_height(box[3])
        
        if legends is not None:
            fig_dict[fig_n].axes.legend(
                patch_dict[fig_n], legends, loc=1,
                prop={'size': 8}, fancybox=True, framealpha=0.5)

    plt.pause(pause)
    plt.draw()


def crop_and_resize(img, center, size, out_size,
                    border_type=cv.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv.INTER_LINEAR):
    """Crop and resize image patch.
    
    Parameters
    ----------
    img : ndarray of shape (size, size)
        Original image
    
    center : ndarray of shape (2,)
        Center of patch to be extracted
    
    size : int
        Size of patch to be extracted
        
    out_size : int
        Size of resized patch
        
    border_type : default=cv.BORDER_CONSTANT
        Type of border when adding padding to image
    
    border_value : ndarray of shape (3,)
        Value to be used for border
    
    interp : default=cv.INTER_LINEAR
        Interpolation method to be used in resizing. 
        cv.INTER_CUBIC would provide higher resolution, but cv.INTER_LINEAR is faster.
    
    Returns
    -------
    patch : ndarray of shape (out_size, out_size)
        Cropped and resized image patch
    """
    # Calculate coordinates of corners (0-indexed)
    size = np.round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size)).astype(int)

    # Pad image (if necessary)
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # Crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # Resize to out_size
    patch = cv.resize(
        patch, 
        (out_size, out_size),
        interpolation=interp
    )

    return patch