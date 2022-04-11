"""Script to train SiamFC network."""
import os
import cv2
#import hydra
import torch
import pytorch_lightning as pl
from got10k.datasets import *
from torch.utils.data import DataLoader
from torchvision import transforms as T
from siamfc import *


# HYPERPARAMETERS
# TODO: Specify hyperparameters in config file (can use Hydra)
batch_size = 8
epoch_num = 50
initial_lr = 1e-2
ultimate_lr = 1e-5
# CONFIGS
# TODO: Include config parameters in config file
# root_dir = '/Users/xiangli/iCloud Drive (Archive)/Desktop/siamfc-pytorch/data/GOT-10k'
root_dir = 'data/GOT-10k'
# root_dir = 'C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/GOT-10k'
pretrained = False
pretrained_alexnet = 'pretrained/siamfc_alexnet_e50.pth'

# For debugging
dataset_opt = 'GOT-10k'


def main():
    # Initialize encoder for SiamFC
    encoder = AlexNet()
    
    # Load pretrained encoder
    if pretrained:
        encoder.load_pretrained(pretrained_alexnet)
    
    # Initialize SiamFC network
    siamfc_model = SiamFCNet(
        encoder=encoder,
        epoch_num = epoch_num,
        batch_size=batch_size,
        initial_lr=initial_lr,
        ultimate_lr=ultimate_lr,
        loss=bce_loss_balanced
    )
    
    # Define transforms
    transforms = SiamFCTransforms()
    
    # Check if GPU available
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Initialize dataloader and train
    if dataset_opt == 'GOT-10k':
        seqs = GOT10k(root_dir=root_dir, subset='train')
        train_dataset = Pair(seqs=seqs, transforms=transforms)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        val_seqs = GOT10k(root_dir=root_dir, subset='val')
        val_dataset = Pair(seqs=val_seqs, transforms=transforms)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=0
        )
    elif dataset_opt == 'imagenet': 
        imagenet = ImageNetDataModule(
            data_dir=root_dir,
            transform=transforms,
            batch_size=batch_size
        )
        
        trainer = pl.Trainer(min_epochs=epoch_num)
        trainer.fit(siamfc_model, datamodule=imagenet)

    # Train model
    trainer = pl.Trainer(
        min_epochs=epoch_num,
        max_epochs=epoch_num,
        accelerator=accelerator,
        devices=1)
    trainer.fit(
        model=siamfc_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()