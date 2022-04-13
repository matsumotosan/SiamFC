"""Script to train SiamFC network."""
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from got10k.datasets import *
from torch.utils.data import DataLoader
from siamfc import *

# For debugging
accelerator = ('gpu' if torch.cuda.is_available() else 'cpu')
dataset_opt = 0


def main(cfg):
    torch.set_default_dtype(torch.float32)
    
    # Initialize encoder
    if cfg.network.arch == 'alexnet':
        encoder = AlexNet()
    elif cfg.network.arch == 'random_walk':
        # encoder = CRW_ResNet()
        pass
    
    # Load pretrained weights (if available)
    if cfg.network.pretrained:
        encoder.load_pretrained(cfg.network.pretrained)
    
    # Initialize SiamFC network
    siamfc_model = SiamFCNet(
        encoder=encoder,
        epoch_num = cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr=cfg.hparams.ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # Define transforms
    transforms = SiamFCTransforms()
    
    # Initialize dataloader
    if dataset_opt == 0:    # GOT-10k
        seqs = GOT10k(root_dir=root_dir, subset='train')
        dataset = Pair(seqs=seqs, transforms=transforms)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=6
        )
        
        val_seqs = GOT10k(root_dir=root_dir, subset='val')
        dataset_val = Pair(seqs=val_seqs, transforms=transforms)
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=6
        )
        
        trainer = pl.Trainer(
            min_epochs=epoch_num,
            max_epochs=epoch_num,
            accelerator=accelerator,
            devices=1
        )
        trainer.fit(
            model=siamfc_model,
            train_dataloaders=dataloader
            #val_dataloaders=dataloader_val
        )
    elif dataset_opt == 1:  # ILSVRC        
        imagenet = ImageNetDataModule(
            data_dir=root_dir,
            transform=transforms,
            batch_size=batch_size
        )
        
        trainer = pl.Trainer(min_epochs=epoch_num)
        trainer.fit(siamfc_model, datamodule=imagenet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracking with SiamFC."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/train/config.yaml",
        help="Path to config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)