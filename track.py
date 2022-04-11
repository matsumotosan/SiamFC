"""Script to run tracking using trained SiamFC network."""
import os
import glob
import numpy as np
import torch
import pytorch_lightning as pl
from siamfc import *
from got10k.trackers import Tracker
# from submodules.videowalk import resnet

response_up = 16
response_sz = 17
scale_lr = 0.35 #0.59
scale_penalty = 0.975
scale_num = 5 #3
exemplar_sz = 127
instance_sz = 255
context = 0.5
window_influence = 0.176

# Hyperparmaters
batch_size = 8
epoch_num = 50
initial_lr = 1e-2
ultimate_lr = 1e-5

# Pre-trained encoder file
encoder_arch = 'alexnet'
pretrained_siamfc_alexnet = 'pretrained/siamfc_alexnet_e50.pth'
pretrained_crw_resnet = 'submodules/videowalk/pretrained.pth'

# Data directory
data_dir = './data/GOT-10k/test/GOT-10k_Test_000150/'

# Notes
# Test 100: struggle to distinguish between two chicks, fast motion
# Test 101: partial occlusion
# Test 150: four skaters, frequently switches between skaters due to similar appearance

# data_dir = 'C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/GOT-10k/train/GOT-10k_Train_000001/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load pre-trained encoder
    if encoder_arch == 'alexnet':
        encoder = AlexNet()
        encoder.load_pretrained(file=pretrained_siamfc_alexnet)
    elif encoder_arch == 'random_walk':
        encoder.load_pretrained(file=pretrained_crw_resnet)
    else:
        raise ValueError('Invalid encoder architecture specified.')
    
    # Initialize SiamFC network
    siamese_net = SiamFCNet(
        encoder=encoder,
        epoch_num = epoch_num,
        batch_size=batch_size,
        initial_lr=initial_lr,
        ultimate_lr = ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # ckpt = torch.load('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    # siamese_net.load_state_dict(ckpt['state_dict'])
    #siamese_net = SiamFCNet.load_from_checkpoint('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    
    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_net
    )
    
    #print(tracker.device)
    # Get data (images and annotations)
    img_files = sorted(glob.glob(data_dir + '*.jpg'))
    anno = np.loadtxt(data_dir + 'groundtruth.txt', delimiter=',')
    
    # Run tracker
    tracker.track(img_files, anno, visualize=True) 
    # tracker.track(img_files, anno[0], visualize=True)


if __name__ == '__main__':
    main()