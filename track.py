"""Tracking using trained SiamFC network."""
import glob
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from siamfc import *
from models import *

# Notes
# Test 100: struggle to distinguish between two chicks, fast motion
# Test 101: partial occlusion
# Test 150: four skaters, frequently switches between skaters due to similar appearance


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pretrained encoder
    encoder, preprocess = load_pretrained_encoder(
        cfg.network.arch, 
        cfg.network.pretrained,
        device)

    # Initialize SiamFC network
    siamese_net = SiamFCNet(
        encoder=encoder,
        epoch_num=cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr=cfg.hparams.ultimate_lr,
        loss=bce_loss_balanced,
        preprocess=preprocess,
        init_weights=False
    )
    
    # ckpt = torch.load('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_9/checkpoints/epoch=49-step=58300.ckpt')
    #ckpt = torch.load('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_14/checkpoints/epoch=49-step=58300.ckpt')
    # siamese_net.load_state_dict(ckpt['state_dict'])
    #siamese_net = SiamFCNet.load_from_checkpoint('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    
    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_net
    )
    
    # Get data (images and annotations)
    img_files = sorted(glob.glob(cfg.data_dir + '*.jpg'))
    anno = np.loadtxt(cfg.data_dir + 'groundtruth.txt', delimiter=',').reshape(-1, 4)
     
    # Run tracker 
    tracker.track(img_files, anno[0], visualize=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/track/track_alexnet.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)
