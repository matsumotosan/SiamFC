#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 00:34:46 2022

@author: xiangli
"""
import os
from got10k.trackers import Tracker
from models import *
from siamfc import *
import numpy as np
from transforms import *
from losses import *
import cv2

response_up = 16
response_sz = 17
scale_step = 1.025 #1.0375
scale_lr = 0.35 #0.59
scale_penalty = 0.975
scale_num = 5 #3
exemplar_sz = 127
instance_sz = 255
context = 0.5
window_influence = 0.176

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


class SiameseFcTracker(Tracker):
    def __init__(self,net_path=None,siamese_net=None):
        super(SiameseFcTracker,self).__init__('SiamFC',True)
        assert siamese_net != None
        self.siamese_net = siamese_net
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.siamese_net.eval()
        if net_path is not None:
            self.siamese_net.load_state_dict(torch.load(net_path))
        self.siamese_net.to(self.device)
        
    @torch.no_grad()
    def init(self,img,box):
        # convert box to 0-indexed and center based [y, x, h, w]
        # Notice that the box given by GOT-10k has format ltwh(left, top, width, height)
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        
        # create hanning window, which is used to regularize the response map 
        self.upscale_sz = response_up*response_sz #This is the size of the response map after upsampling
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        
        # search scale factors
        self.scale_factors = scale_step ** np.linspace(
            -(scale_num // 2),
          scale_num // 2, scale_num)
        
        # exemplar and search sizes
        margin = context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + margin))
        self.x_sz = self.z_sz * \
            instance_sz / exemplar_sz
        
        #get the exemplar image from the first frame
        self.avg_color = np.mean(img,axis=(0,1))
        z = crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        #get the deep feature for the exemplar image
        z = torch.from_numpy(z).to(self.device).permute(2,0,1).unsqueeze(0).float()
        self.kernel = self.siamese_net.encoder(z) #size: 1x1x17x17
    
    @torch.no_grad()
    def update(self,img):
        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=instance_sz,
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
        responses[:scale_num // 2] *= scale_penalty
        responses[scale_num // 2 + 1:] *= scale_penalty
        
        #choose the response map with the largest peak value
        scale_id = np.argmax(np.amax(responses,axis=(1,2)))
        
        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        #Apply hanning window to the response map
        response = (1 - window_influence) * response + \
            window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.siamese_net.total_stride / response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - scale_lr) * 1.0 + \
            scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self,img_files,box,visualize = False):
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
    
if __name__ == '__main__':
    encoder = AlexNet()
    batch_size = 8
    epoch_num = 50
    lr = 1e-2
    siamese_net = SiamFCNet(
        encoder=encoder,
        batch_size=batch_size,
        lr=lr,
        loss=bce_loss_balanced
    )
    tracker = SiameseFcTracker(siamese_net = siamese_net)
    
    seq_dir = os.path.expanduser('/Users/xiangli/iCloud Drive (Archive)/Desktop/siamfc-pytorch/data/GOT-10k/train/GOT-10k_Train_000001')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth.txt')
    
    #net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker.track(img_files, anno[0], visualize=True)
    
    