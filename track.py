"""Run SiamFC tracker using pretrained encoder."""
import glob
import numpy as np
import torch
from siamfc import *
from .submodules.videowalk import resnet


# Tracker settings
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

# Hyperparmaters
batch_size = 8
epoch_num = 50
lr = 1e-2

# Pre-trained encoder file
encoder_arch = 'alexnet'
pretrained_siamfc_alexnet = 'pretrained/siamfc_alexnet_e50.pth'
pretrained_crw_resnet = 'submodules/videowalk/pretrained.pth'

# Data directory
data_dir = './data/GOT-10k/train/GOT-10k_Train_000040/'
device = torch.device('cpu')


def main():
    # Load pre-trained encoder
    if encoder_arch == 'alexnet':
        encoder = AlexNet()
        encoder.load_pretrained(file=pretrained_siamfc_alexnet)
    elif encoder_arch == 'random_walk':
        encoder.load_pretrained(file=pretrained_crw_resnet)
    else:
        raise ValueError('Invalid encoder architecture specified.')
    
    # Initialize SiamFC network and set to .eval() mode
    siamese_model = SiamFCNet(
        encoder=encoder,
        batch_size=batch_size,
        lr=lr,
        loss=bce_loss_balanced
    )
    siamese_model.eval()
    
    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_model
    )
    
    # Get data (images and annotations)
    img_files = sorted(glob.glob(data_dir + '*.jpg'))
    anno = np.loadtxt(data_dir + 'groundtruth.txt', delimiter=',')
    
    # Run tracker
    tracker.track(img_files, anno[0], visualize=True)
    

if __name__ == '__main__':
    main()