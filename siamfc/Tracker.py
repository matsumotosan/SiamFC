import os
import cv2
import time
import glob
import torch
import numpy as np
from got10k.trackers import Tracker
from siamfc.utils  import crop_and_resize, read_image, show_image
from collections import namedtuple

# Tracker configurations
response_up = 16
response_sz = 17
scale_step = 1.025 #1.0375
scale_lr = 0.35 #0.59
scale_penalty = 0.975
scale_num = 5 #3
exemplar_sz = 127
instance_sz = 255
context = 0.5


class SiamFCTracker(Tracker):
    def __init__(self, net_path=None, siamese_net=None, **kwargs):
        super().__init__('SiamFC', True)
        assert siamese_net != None
        
        self.siamese_net = siamese_net
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.siamese_net.eval()
        
        if net_path is not None:
            self.siamese_net.encoder.load_state_dict(torch.load(net_path,map_location=self.device))
        self.siamese_net.to(self.device)
        self.cfg = self.parse_args(**kwargs)
        
    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            #'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3, #5
            'scale_step': 1.025, #1.0375,
            'scale_lr': 0.35, #0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            #'total_stride': 8,
            # train parameters
            #'epoch_num': 50,
            #'batch_size': 8,
            #'num_workers': 32,
            #initial_lr': 1e-2,
            #'ultimate_lr': 1e-5,
            #'weight_decay': 5e-4,
            #'momentum': 0.9,
            #'r_pos': 16,
            #'r_neg': 0
            }
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # convert box to 0-indexed and center based [y, x, h, w]
        # Notice that the box given by GOT-10k has format ltwh(left, top, width, height)
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        
        # create hanning window, which is used to regularize the response map 
        self.upscale_sz = self.cfg.response_up*self.cfg.response_sz #This is the size of the response map after upsampling
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        
        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
          self.cfg.scale_num // 2, self.cfg.scale_num)
        
        # exemplar and search sizes
        margin = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + margin))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        #get the exemplar image from the first frame
        self.avg_color = np.mean(img,axis=(0,1))
        z = crop_and_resize(
            img, self.center, self.z_sz,
            out_size = self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        #get the deep feature for the exemplar image
        z = torch.from_numpy(z).to(self.device).permute(2,0,1).unsqueeze(0).float()
        self.kernel = self.siamese_net.encoder(z) #size: 1x1x17x17
    
    @torch.no_grad()
    def update(self, img):
        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        #compute deep features for x
        x = self.siamese_net.encoder(x) 
        
        #compute response map
        responses = self.siamese_net._xcorr(self.kernel,x) #size: (scale_num x 1 x 17 x 17)
        responses = responses.squeeze(1).cpu().numpy() #size: (scale_num x 17 x 17)
        
        #upsample responses and penalizae scale change
        #responses has size: (scale_num x 272 x 272 )
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses]) 
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty
        
        #choose the response map with the largest peak value
        scale_id = np.argmax(np.amax(responses,axis=(1,2)))
        
        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        #Apply hanning window to the response map
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.siamese_net.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num,4))
        boxes[0] = box
        times = np.zeros(frame_num)
        
        for f, img_file in enumerate(img_files):
            img = read_image(img_file)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times