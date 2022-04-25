"""Transforms for GOT-10k dataset. 
Adapted from https://github.com/huanglianghua/siamfc-pytorch."""
import cv2 as cv
import numpy as np
import numbers
import torch
from torchvision import transforms
from .utils import crop_and_resize


class RandomStretch:
    """Randomly stretch an image"""
    def __init__(self, max_stretch: float = 0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img):
        interp = np.random.choice([
            cv.INTER_LINEAR,
            cv.INTER_CUBIC,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_LANCZOS4])
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale))
        return cv.resize(img, out_size, interpolation=interp)


class CenterCrop(object):
    """Crop image with specified size at the center of image."""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            img = cv.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad
        patch = img[i:i+th,j:j+tw]
        return patch


class RandomCrop:
    """Randomly crop a patch of specified size from image."""
    def __init__(self, size: int):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


class SiamFCTransforms:
    """Transforms for SiamFC exemplar and search images."""
    def __init__(
        self,
        exemplar_sz: int = 127, 
        instance_sz: int = 255, 
        context: float = 0.5
    ):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transform_z = transforms.Compose([
            # RandomStretch(),
            CenterCrop(exemplar_sz),
            transforms.ToTensor()
        ])
        self.transform_x = transforms.Compose([
            # RandomStretch(),
            CenterCrop(instance_sz),
            transforms.ToTensor()
        ])

    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz)
        x = self._crop(x, box_x, self.instance_sz)
        z = self.transform_z(z)
        x = self.transform_x(x)
        return z, x

    def _crop(self, img, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w].
        # Notice that the box given by GOT-10k has format ltwh(left, top, width, height)
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv.INTER_LINEAR,
            cv.INTER_CUBIC,
            cv.INTER_AREA,
            cv.INTER_NEAREST,
            cv.INTER_LANCZOS4])
        patch = crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        
        return patch