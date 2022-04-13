"""Script to run tracking using trained SiamFC network."""
import os
import glob
import torch
import yaml
import argparse
import numpy as np
import pytorch_lightning as pl
from siamfc import *
from omegaconf import OmegaConf

# Pre-trained encoder file
encoder_arch = 'alexnet'
pretrained_siamfc_alexnet = 'pretrained/siamfc_alexnet_e50.pth'
pretrained_crw_resnet = 'submodules/videowalk/pretrained.pth'

# Data directory
# data_dir = './data/GOT-10k/train/GOT-10k_Train_000001/'
data_dir = './data/GOT-10k/test/GOT-10k_Test_000150/'
# data_dir = 'C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/GOT-10k/train/GOT-10k_Train_000001/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Notes
# Test 100: struggle to distinguish between two chicks, fast motion
# Test 101: partial occlusion
# Test 150: four skaters, frequently switches between skaters due to similar appearance


def main(cfg):
    # Load pre-trained encoder
    if cfg.network.arch == 'alexnet':
        encoder = AlexNet()
        encoder.load_pretrained(file=pretrained_siamfc_alexnet)
    elif cfg.network.arch == 'random_walk':
        encoder.load_pretrained(file=pretrained_crw_resnet)
    
    # Initialize SiamFC network
    siamese_net = SiamFCNet(
        encoder=encoder,
        epoch_num = cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr = cfg.hparams.ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # ckpt = torch.load('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    # siamese_net.load_state_dict(ckpt['state_dict'])
    # siamese_net = SiamFCNet.load_from_checkpoint('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    
    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_net
    )
    
    #print(tracker.device)
    # Get data (images and annotations)
    img_files = sorted(glob.glob(data_dir + '*.jpg'))
    anno = np.loadtxt(data_dir + 'groundtruth.txt', delimiter=',')
    
    # Run tracker
    # tracker.track(img_files, anno[0], visualize=True)   # training videos
    tracker.track(img_files, anno, visualize=True)    # test videos


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/config.yaml",
        help="Path to config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)