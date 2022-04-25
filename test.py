import argparse
from omegaconf import OmegaConf
from got10k.experiments import *
import glob
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from siamfc import *
from siamfc.models import *

def main_alex_net(cfg):
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
        name='alex_pretrained_Got10k',
        **cfg.tracker
    )
    
    e = ExperimentOTB('C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/OTB',version=2015)
    e.run(tracker,visualize=False)
    e.report([tracker.name])

def main_alex_net_imageNet(cfg):
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
        name='alex_Image_Net_pretrained_Got10k',
        **cfg.tracker
    )
    
    e = ExperimentOTB('C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/OTB',version=2015)
    e.run(tracker,visualize=False)
    e.report([tracker.name])


def main_res_net(cfg):
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
        name='resnet_Got10k',
        **cfg.tracker
    )
    
    e = ExperimentOTB('C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/OTB',version=2015)
    e.run(tracker,visualize=False)
    e.report([tracker.name])

    
def main_res_net_ImageNet(cfg):
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
        name='resnet_Got10k_ImageNet',
        **cfg.tracker
    )
    
    e = ExperimentOTB('C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/OTB',version=2015)
    e.run(tracker,visualize=False)
    e.report([tracker.name])


if __name__ == "__main__":
    '''
    Test alexnet pretrained on got10k
    '''
    
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
        
    main_alex_net(cfg)
    
    
    '''
    Test alexnet using imagenet_pretrained weights
    '''
    
    
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/track/track_alexnet-torch.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
    
    main_alex_net_imageNet(cfg)
    
    
    '''
    Test alexnet using resnet 18 trained on GOT-10k
    '''
    
    
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/track/track_resnet18.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
    
    main_res_net(cfg)
    
    
    '''
    Test alexnet using resnet 18 trained on ImageNet
    '''
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/track/track_resnet18_ImageNet.yaml",
        help="Path to tracking config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
    
    main_res_net_ImageNet(cfg)