"""Track using trained SiamFC network."""
import glob
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from siamfc import *
from siamfc.models import *

# Notes
# Test 100: struggle to distinguish between two chicks, fast motion
# Test 101: partial occlusion
# Test 150: four skaters, frequently switches between skaters due to similar appearance


def main(cfg):
    # Load pretrained encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, preprocess = load_pretrained_encoder(
        cfg.network.arch,
        cfg.network.pretrained,
        device
    )

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
    siamese_net.eval()

    # Initialize tracker
    tracker = SiamFCTracker(
        siamese_net=siamese_net,
        **cfg.tracker
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
        default="./conf/track/track_resnet18-crw.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)