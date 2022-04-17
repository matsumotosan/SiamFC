"""Train SiamFC network."""
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from got10k.datasets import GOT10k
from siamfc import *
from models import *


def main(cfg):
    torch.set_default_dtype(torch.float32)
    
    # Initialize encoder
    if cfg.network.arch == 'alexnet':
        encoder = AlexNet()
        # encoder = AlexNet_torch()
    # elif cfg.network.arch == 'random_walk':
    #     encoder = ResNet()
    
    # Initialize SiamFC network
    siamfc_model = SiamFCNet(
        encoder=encoder,
        epoch_num=cfg.hparams.epoch_num,
        batch_size=cfg.hparams.batch_size,
        initial_lr=cfg.hparams.initial_lr,
        ultimate_lr=cfg.hparams.ultimate_lr,
        weight_decay=cfg.hparams.weight_decay,
        loss=bce_loss_balanced,
        init_weights=True
    )
    
    # Load from last checkpoint (if available)
    # siamfc_model.load_from_checkpoint()
    
    # Define transforms
    transforms = SiamFCTransforms()
    
    # Initialize dataloaders
    if cfg.data.name == "got10k":
        # got10k_dm = GOT10kDataModule()
        
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

    # Setup logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=cfg.network.arch
    ) 

    # Setup checkpoint (improvement in validation loss)
    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_losss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=-1
    )
    
    # Setup checkpoint (latest checkpoint)
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=500,
        save_top_k=1
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=cfg.hparams.epoch_num,
        max_epochs=cfg.hparams.epoch_num,
        callbacks=[val_checkpoint, latest_checkpoint],
        accelerator="auto",
        devices="auto",
        logger=logger
    ) 

    # Train model
    trainer.fit(
        model=siamfc_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Test model
    trainer.test(
        model=siamfc_model,
        test_dataloaders=test_dataloader
    )
    
    # trainer.fit(
    #     model=siamfc_model,
    #     datamodule=got10k_dm
    # )
    
    # Save encoder weights
    torch.save(
        siamfc_model.encoder.state_dict(),
        cfg.network.trained_model_path
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