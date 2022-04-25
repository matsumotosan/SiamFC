"""Test tracker performance on various datasets."""
import argparse
import torch
from omegaconf import OmegaConf
from siamfc import *
from siamfc.models import *
from got10k.experiments import *


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
        name=cfg.network.arch,
        **cfg.tracker
    )
    
    # Run experiments
    tracker.name = cfg.network.name
    exp = ExperimentOTB(cfg.data_dir, version=2015)
    exp.run(tracker, visualize=False)
    exp.report([tracker.name])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/test/test.yaml",
        help="Path to testing config file."
    )

    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)

    main(cfg)

    # for i in range(len(cfg.network)):
    #     cfg_i = cfg
    #     OmegaConf.update(cfg_i, "network", cfg.network[i], merge=False)
    #     main(cfg_i)