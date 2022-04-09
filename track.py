import os
import cv2
import glob
import numpy as np
from siamfc import *
from got10k.trackers import Tracker


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


def main():
    # Initialize SiamFC
    encoder = AlexNet()
    siamese_net = SiamFCNet(
        encoder=encoder,
        batch_size=batch_size,
        lr=lr,
        loss=bce_loss_balanced
    )
    
    # Initialize tracker
    tracker = SiamFCTracker(siamese_net=siamese_net)
    
    # Get data
    seq_dir = os.path.expanduser('/Users/xiangli/iCloud Drive (Archive)/Desktop/siamfc-pytorch/data/GOT-10k/train/GOT-10k_Train_000001')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth.txt')
    
    #net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker.track(img_files, anno[0], visualize=True)


if __name__ == '__main__':
    main()