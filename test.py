import argparse
from omegaconf import OmegaConf
from got10k.experiments import *


def main(cfg):
    pass


if __name__ == "__main__":
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