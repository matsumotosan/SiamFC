"""Tracker class for SiamFC architecture. Based on the GOT-10k toolkit Tracker class at
https://github.com/got-10k/toolkit/blob/master/got10k/trackers/__init__.py."""
import time
import torch
import cv2 as cv
import numpy as np
from siamfc.utils  import crop_and_resize, read_image, show_image
from collections import namedtuple


class SiamFCTracker:
    """Tracking head for SiamFC network.
    
    Parameters
    ----------
    siamese_net : nn.Module
        SiamFC network with pretrained weights.
    """
    def __init__(self, siamese_net, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.siamese_net = siamese_net
        self.siamese_net.eval()
        self.siamese_net.to(self.device)
        self.cfg = self.parse_args(**kwargs)
        
    def parse_args(self, **kwargs):
        # Default parameters
        cfg = {
            'exemplar_sz': 127,
            'instance_sz': 255,
            'response_sz': 17,
            'upsample_factor': 16,
            'context': 0.5,
            'scale_num': 3,
            'scale_step': 1.025,
            'scale_lr': 0.35,
            'scale_penalty': 0.975,
            'window_influence': 0.176,
        }
        
        # Update parameters
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    def track(self, img_files, box, visualize=False):
        """Track given set of frames given initial box. Optionally visualize tracking.
        
        Parameters
        ----------
        img_files : list (frames,)
            Ordered list of files to frames
        
        box : ndarray of shape (4,)
            Initial box
        
        visualize : bool, default=False
            Visualize tracking if True
        
        Returns
        -------
        boxes : ndarray of shape (frames, 4)
            Bounding box for object at each frame

        t : ndarray of shape (frames,)
            Time stamps
        """
        # Get total number of frames
        n_frames = len(img_files)
        
        # Initialize tracking box for each frame
        boxes = np.zeros((n_frames, 4))
        boxes[0] = box
        t = np.zeros(n_frames)
        response = None
        
        # Iterate through each frame
        for frame, img_file in enumerate(img_files):
            img = read_image(img_file)
            t0 = time.time()
            if frame == 0:
                self.init(img, boxes[0])
            else:
                boxes[frame, :], response = self.update(img)
                
            t[frame] = time.time() - t0
            if visualize:
                show_image(img, boxes[frame, :], response_map=response)
                
        return boxes, t
    
    @torch.no_grad()
    def init(self, img, box, box_style='ltwh') -> None:
        """Initialize tracking box.
        
        Parameters
        ----------
        img :
            Initial image
        
        box :
            Initial box
            
        box_style : str, default='ltwh'
            Annotation format of box dimensions
        """
        # Convert box format to be 0-indexed and center based [y, x, h, w]
        if box_style == 'ltwh':
            box = self._ltwh2yxhw(box)
            self.box_center = box[:2]
            self.box_sz = box[2:]
        
        # Calculate size of upsampled response map
        self.upsample_sz = self.cfg.upsample_factor * self.cfg.response_sz
        
        # Create hanning window to regularize response map
        self._create_hanning_window()
        
        # Calculate scale factors
        self._get_scale_factors()
        
        # Calculate image scaling
        p = self.cfg.context * np.mean(self.box_sz)
        self.z_sz = np.sqrt(np.prod(self.box_sz + 2 * p))
        self.x_sz = self.z_sz * (self.cfg.instance_sz / self.cfg.exemplar_sz)
        
        # Get exemplar image from first frame
        self.avg_color = np.mean(img, axis=(0, 1))
        z = crop_and_resize(
            img, 
            self.box_center, 
            self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color
        )
        
        # Get exemplar image embedding
        z = torch.from_numpy(z).to(self.device).permute(2,0,1).unsqueeze(0).float()
        if self.siamese_net.preprocess == True:
            z /= 255
            z = self.siamese_net.normalize(z)
        self.kernel = self.siamese_net.encoder(z)
    
    @torch.no_grad()
    def update(self, img):
        """Update tracker given new frame.
        
        Parameters
        ----------
        img : ndarray of shape (H, W, 3)
            Next frame to be tracked
        
        Returns
        -------
        box : ndarray of shape (4,)
            Bounding box for new frame
        """
        # Get search images at different scales
        x = [crop_and_resize(
                img, 
                self.box_center, 
                in_size=self.x_sz * f,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color) 
            for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        if self.siamese_net.preprocess:
            x /= 255
            x = self.siamese_net.normalize(x)
            
        # Compute search image embedding
        x = self.siamese_net.encoder(x) 
       
        # Compute response map (scale_num, 1, 17, 17)
        responses = self.siamese_net._xcorr(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy() # size: (scale_num x 17 x 17)
        
        # Upsample response map (17, 17) -> (272, 272)
        responses = np.stack([cv.resize(r, 
            (self.upsample_sz, self.upsample_sz),
            interpolation=cv.INTER_CUBIC)
            for r in responses])
        
        # Penalize scale change
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        
        # Choose response map with largest peak value
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        response = responses[scale_id]
        
        # Normalize response map
        response -= response.min()
        response /= response.sum() + 1e-16
        
        # Apply hanning window to the response map
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hanning_window
        loc = np.unravel_index(response.argmax(), response.shape)
        
        # Locate target center
        disp_in_response = np.array(loc) - (self.upsample_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.siamese_net.total_stride / self.cfg.upsample_factor
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.box_center += disp_in_image

        # Update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
            
        self.box_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale
        
        # Return 1-indexed and left-top based bounding box
        box = np.array([
            self.box_center[1] + 1 - (self.box_sz[1] - 1) / 2,
            self.box_center[0] + 1 - (self.box_sz[0] - 1) / 2,
            self.box_sz[1], self.box_sz[0]])
        
        return box, response

    def _ltwh2yxhw(self, box):
        box = np.array([
            box[1] + (box[3] - 1) / 2 - 1,
            box[0] + (box[2] - 1) / 2 - 1,
            box[3], box[2]],
            dtype=np.float32)
        return box
    
    def _create_hanning_window(self) -> None:
        self.hanning_window = np.outer(
            np.hanning(self.upsample_sz),
            np.hanning(self.upsample_sz))
        self.hanning_window /= self.hanning_window.sum()
        
    def _get_scale_factors(self):
        powers = np.linspace(
            -(self.cfg.scale_num // 2), 
            self.cfg.scale_num // 2, 
            self.cfg.scale_num
        )
        self.scale_factors = self.cfg.scale_step ** powers