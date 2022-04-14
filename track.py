"""Script to run tracking using trained SiamFC network."""
import glob
import argparse
import numpy as np
from omegaconf import OmegaConf
from siamfc import *
from submodules.videowalk.code import resnet

# Notes
# Test 100: struggle to distinguish between two chicks, fast motion
# Test 101: partial occlusion
# Test 150: four skaters, frequently switches between skaters due to similar appearance


def main(cfg):
    # Load pretrained encoder
    if cfg.network.arch == 'alexnet':
        encoder = AlexNet()
        encoder.load_pretrained(file=cfg.network.pretrained)
    elif cfg.network.arch == 'resnet18':
        encoder = resnet.resnet18(pretrained=cfg.network.pretrained)
    elif cfg.network.arch == 'resnet50':
        encoder = resnet.resnet50(pretrained=cfg.network.pretrained)
    else:
        raise ValueError('Invalid network architecture specified.')
    
    # Initialize SiamFC network
    siamese_net = SiamFCNet(
        encoder=encoder,
        epoch_num = cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr=cfg.hparams.ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # ckpt = torch.load('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    # siamese_net.load_state_dict(ckpt['state_dict'])
    # siamese_net = SiamFCNet.load_from_checkpoint('C:/Users/xw/Desktop/eecs 542 final project/SiamFC-master/lightning_logs/version_0/checkpoints/epoch=3-step=4664.ckpt')
    
    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_net
    )
    
    # Get data (images and annotations)
    img_files = sorted(glob.glob(cfg.data_dir + '*.jpg'))
    anno = np.loadtxt(cfg.data_dir + 'groundtruth.txt', delimiter=',')
    
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
        default="./conf/track/track_alexnet.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)