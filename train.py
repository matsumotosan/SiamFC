#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from got10k.datasets import*
import os
import cv2
from transform import *
from siamfc import *
from models import *
from datasets import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
batch_size = 8
epoch_num = 50

if __name__ == '__main__':
    root_dir = '/Users/xiangli/iCloud Drive (Archive)/Desktop/siamfc-pytorch/data/GOT-10k'
    seqs = GOT10k(root_dir,subset='train',return_meta=True)
    SiamFC = SiamFCNet(AlexNet())
    transforms = SiamFCTransforms()
    # setup the dataset
    dataset = Pair(
        seqs = seqs,
        transforms = transforms
        )
    # setup the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True
        )
    trainer = pl.Trainer()
    # start training
    trainer.fit(model=SiamFC,train_dataloaders = dataloader)