"""Script to train SiamFC network."""
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import OmegaConf
from got10k.datasets import GOT10k
from siamfc import *


def main(cfg):
    accelerator = ('gpu' if torch.cuda.is_available() else 'cpu')
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
        epoch_num=cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr=cfg.hparams.ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # Define transforms
    transforms = SiamFCTransforms()
    
    # Initialize dataloaders
    if cfg.data.name == "got10k":
        got10k_dm = GOT10kDataModule()
        
        # Training
        seqs = GOT10k(root_dir=cfg.data.root_dir, subset='train')
        train_dataset = Pair(seqs=seqs, transforms=transforms)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.hparams.batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=6
        )
        
        # Validation
        val_seqs = GOT10k(root_dir=cfg.data.root_dir, subset='val')
        val_dataset = Pair(seqs=val_seqs, transforms=transforms)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=cfg.hparams.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=6
        )
        
        # Test
        test_seqs = GOT10k(root_dir=cfg.data.root_dir, subset='test')
        test_dataset = Pair(seqs=test_seqs, transforms=transforms)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.hparams.batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=6
        )
        
    elif cfg.data.name == "imagenet":    
        imagenet = ImageNetDataModule(
            data_dir=cfg.data.root_dir,
            transform=transforms,
            batch_size=cfg.hparams.batch_size
        )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=cfg.hparams.epoch_num,
        max_epochs=cfg.hparams.epoch_num,
        accelerator=accelerator,
        devices=1
    )
    
    # # Train model
    # trainer.fit(
    #     model=siamfc_model,
    #     train_dataloaders=train_dataloader
    #     # val_dataloaders=val_dataloader,
    #     # test_dataloaders=test_dataloader
    # )
    
    trainer.fit(
        model=siamfc_model,
        datamodule=got10k_dm
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training SiamFC network."
    )
    parser.add_argument(
        "--config",
        dest="config_file", 
        default="./conf/train/train_alexnet.yaml",
        help="Path to training config file."
    )
    
    args = parser.parse_args()
    with open(args.config_file) as f:
        cfg = OmegaConf.load(f)
        
    main(cfg)