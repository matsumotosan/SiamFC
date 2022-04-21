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
    
    @staticmethod
    def parse_args(**kwargs):
        # Default parameters
        cfg = {
            'exemplar_sz': 127,
            'instance_sz': 255,
            'score_sz': 17,
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
    
    def track(self, img_files, box, box_style='ltwh', visualize=True, video_name=""):
        """Track given set of frames given initial box. Optionally visualize tracking.
        
        Parameters
        ----------
        img_files : list (n_frames,)
            List of image files
        
        box : ndarray of shape (4,)
            Initial bounding box
        
        visualize : bool, default=True
            Visualize tracking if True
        
        video_name : str
            Name of video to be used in window title
        
        Returns
        -------
        boxes : ndarray of shape (frames, 4)
            Bounding box for object at each frame

        t : ndarray of shape (frames,)
            Time stamps
        """
        # Get total number of frames
        n_frames = len(img_files)

        # Initialize tracking parameters
        img = read_image(img_files[0])
        self.init(img, box, box_style)
        if visualize:
            self.display(img, box, window_title=f"{video_name} Frame {1}/{n_frames}")

        # Update tracker for each frame
        for frame, img_file in enumerate(img_files[1:]):
            img = read_image(img_file)
            box, score_map, search_img = self.update(img)
            if visualize:
                self.display(
                    img,
                    box,
                    score_map=score_map,
                    search_img=search_img,
                    window_title=f"{video_name} Frame {frame+2}/{n_frames}"
                )

        cv.destroyAllWindows()
    
    @staticmethod
    def display(img, box, score_map=None, search_img=None, window_title=""):
        show_image(
            img,
            box,
            score_map=score_map,
            search_img=search_img,
            window_title=window_title
        )
    
    @torch.no_grad()
    def init(self, img, box, box_style='ltwh') -> None:
        """Initialize tracker parameters. Pre-calculates kernel (exemplar image embedding).
        
        Parameters
        ----------
        img : ndarray of shape (H, W, 3)
            Initial frame
        
        box : ndarray of shape (4,)
            Initial bounding box coordinates
            
        box_style : str, default='ltwh'
            Annotation format of bounding box
        """
        # Convert box format to be 0-indexed and center based [y, x, h, w]
        if box_style == 'ltwh':
            box = self._ltwh2yxhw(box)

        # Initialize bounding box and score map
        self.box_center = box[:2]
        self.box_sz = box[2:]
        self.score_map = None

        # Calculate size of upsampled score map
        self.upsample_sz = self.cfg.upsample_factor * self.cfg.score_sz

        # Create hanning window to regularize score map
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

        # Reshape (H, W, 3) -> (1, 3, H, W)
        z = torch.from_numpy(z).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

        # Normalize (if necessary)
        if self.siamese_net.preprocess == True:
            z /= 255
            z = self.siamese_net.normalize(z)

        # Calculate exemplar image embedding
        self.kernel = self.siamese_net.encoder(z)

    @torch.no_grad()
    def update(self, img, interp=cv.INTER_CUBIC):
        """Update tracker given new frame. Calculate cross correlation score map for 
        single target embedding and search embeddings over different scales.
        
        Parameters
        ----------
        img : ndarray of shape (H, W, 3)
            Next frame to be tracked
        
        interp : cv.INTERPOLATIONFLAGS, default=cv.INTER_CUBIC
            Upsampling interpolation method
        
        Returns
        -------
        box : ndarray of shape (4,)
            Bounding box for new frame
        
        score_map : ndarray of shape ()
            Cross correlation score map
        """
        # Get search images at different scales
        search_images = np.stack([crop_and_resize(
            img,
            self.box_center,
            in_size=self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) 
                                  for f in self.scale_factors], axis=0
        )
        
        # Reshape (n_scales, H, W, 3) -> (n_scales, 3, H, W)
        x = torch.from_numpy(search_images).to(self.device).permute(0, 3, 1, 2).float()
        
        # Normalize (if necessary)
        if self.siamese_net.preprocess:
            x /= 255
            x = self.siamese_net.normalize(x)

        # Calculate and choose best score map over all search image scales
        score_map, score_idx, loc = self._calculate_score_map(
            self.siamese_net.encoder(x),
            interp=interp
        )

        # Locate target center
        disp_in_response = loc - (self.upsample_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.siamese_net.total_stride / self.cfg.upsample_factor
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[score_idx] / self.cfg.instance_sz
            
        # Update bounding box center
        self.box_center += disp_in_image

        # Update size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[score_idx]
            
        self.box_sz *= scale    # bounding bbox
        self.z_sz *= scale      # exemplar image
        self.x_sz *= scale      # search image
        
        # Return 1-indexed and left-top based bounding box
        box = np.array([
            self.box_center[1] + 1 - (self.box_sz[1] - 1) / 2,  # y
            self.box_center[0] + 1 - (self.box_sz[0] - 1) / 2,  # x
            self.box_sz[1],                                     # w
            self.box_sz[0]                                      # h
        ])
        return box, score_map, search_images[score_idx]

    @torch.no_grad() 
    def _calculate_score_map(self, x, interp):
        # Compute score map
        scores = self.siamese_net._xcorr(self.kernel, x) # (scale_num, 1, 17, 17)
        scores = scores.squeeze(1).cpu().numpy()         # (scale_num x 17 x 17)
        
        # Upsample each score map (17, 17) -> (272, 272)
        scores = np.stack([
            cv.resize(
                score, 
                (self.upsample_sz, self.upsample_sz), 
                interpolation=interp
            ) for score in scores]
        )
        
        # Penalize scale change
        scores[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        scores[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        
        # Choose score map with largest peak value
        score_map, score_idx = self._get_best_score_map(scores)
        
        # Normalize score map
        score_map -= score_map.min()
        score_map /= score_map.sum() + 1e-16
        
        # Apply hanning window to the score map
        score_map = (1 - self.cfg.window_influence) * score_map + \
            self.cfg.window_influence * self.hanning_window

        # Calculate maximum score location
        loc = np.array(np.unravel_index(score_map.argmax(), score_map.shape))

        return score_map, score_idx, loc

    def _create_hanning_window(self):
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
    
    @staticmethod
    def _ltwh2yxhw(box):
        box = np.array([
            box[1] + (box[3] - 1) / 2 - 1,
            box[0] + (box[2] - 1) / 2 - 1,
            box[3], box[2]],
            dtype=np.float32)
        return box
    
    @staticmethod
    def _get_best_score_map(scores):
        score_idx = np.argmax(np.amax(scores, axis=(1, 2)))
        score_map = scores[score_idx]
        return score_map, score_idx
