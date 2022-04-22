"""Transforms for GOT-10k dataset. 
Adapted from https://github.com/huanglianghua/siamfc-pytorch."""
from ctypes.wintypes import SIZE
import cv2
import numpy as np
import numbers
import torch
from torchvision import transforms


class Compose:
    """Class to compose multiple transforms into one."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch:
    """Randomly stretch an image"""
    def __init__(self, max_stretch: float = 0.05):
        self.max_stretch = max_stretch
    
    def __call__(self, img):
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale))
        return cv2.resize(img, out_size, interpolation=interp)


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
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad
        patch = img[i:i+th,j:j+tw]
        #Make sure the size of patch is correct
        offset = np.sum(np.abs(np.array(patch.shape[0:2])-np.array(self.size))) #There might be some small pdifference between the size of the patch and the ideal size, the offset shouldnt be too large
        assert offset <= 4
        patch = cv2.resize(patch,self.size)
        '''
        #Uncomment the following lines to check the exemplar image and the search region. 
        #Notice that many training samples include padding regions, which may cause some implicit bias.
        cv2.imshow('display',patch)
        cv2.waitKey(0)
        '''
        #print(self.size,patch.shape[0:2])
        assert patch.shape[0:2] == self.size
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

class ToTensor:
    """Convert numpy array to Torch tensor."""
    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2,0,1))


# class SiamFCDataTransform:
#     """Transforms for SiamFC exemplar and instance images."""
#     def __init__(self, exemplar_sz: int, instance_sz: int, context: float) -> None:
#         self.exemplar_sz = exemplar_sz
#         self.instance_sz = instance_sz
#         self.context = context
        
#         self.transform_instance = transforms.Compose([
#             transforms.Resize(),
#             transforms.CenterCrop(),
#             transforms.ToTensor()
#         ])
#         self.transform_instance = transforms.Compose([
#             transforms.Resize(),
#             transforms.CenterCrop(),
#             transforms.ToTensor()
#         ])
    
#     def __call__(self, z, x):
#         z = self.transform_instance(z)
#         x = self.transform_exemplar(x)
#         return z, x


class SiamFCTransforms:
    """Transforms for SiamFC exemplar and search images."""
    def __init__(
        self,
        exempler_sz: int = 127, 
        instance_sz: int = 255, 
        context: float = 0.5
    ):
        self.exemplar_sz = exempler_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transform_z = transforms.Compose([
            RandomStretch(),
            CenterCrop(exempler_sz),
            ToTensor()
        ])
        self.transform_x = transforms.Compose([
            RandomStretch(),
            CenterCrop(instance_sz),
            ToTensor()
        ])
    
    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz)
        x = self._crop(x, box_x, self.instance_sz)
        z = self.transform_z(z)
        x = self.transform_x(x)
        return z, x
    
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
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        
        return patch
    

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