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
#root_dir = 'data/GOT-10k'
root_dir = 'C:/Users/xw/Desktop/tracking restart/siamfc-pytorch/data/GOT-10k'
pretrained = False
pretrained_alexnet = 'pretrained/siamfc_alexnet_e50.pth'

# For debugging
dataset_opt = 0


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
    
    # Initialize dataloader
    if dataset_opt == 0:    # GOT-10k
        seqs = GOT10k(root_dir=root_dir, subset='train')
        dataset = Pair(seqs=seqs, transforms=transforms)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        val_seqs = GOT10k(root_dir=root_dir, subset='val')
        dataset_val = Pair(seqs=val_seqs, transforms=transforms)
        dataloader_val = DataLoader(
            dataset_val, 
            batch_size=batch_size, 
            shuffle=False,
            drop_last=True,
            num_workers=0
        )
        
        trainer = pl.Trainer(min_epochs=epoch_num,max_epochs=epoch_num,accelerator="gpu",devices=1)
        trainer.fit(
            model=siamfc_model,
            train_dataloaders=dataloader,
            val_dataloaders=dataloader_val
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
    main()