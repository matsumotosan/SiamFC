"""Train SiamFC network."""
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from siamfc import *
from siamfc.models import *


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

    # Initialize GOT-10k datamodule
    got10k_dm = GOT10kDataModule(
        data_dir=cfg.data.root_dir,
        batch_size=cfg.hparams.batch_size
    )

    # Initialize logger
    logger = setup_logger(
        logger=cfg.logging.logger,
        save_dir=cfg.logging.log_dir,
        name=cfg.network.arch
    )

    # Initialize checkpoint callbacks
    checkpoints = setup_checkpoints(
        checkpoints=cfg.logging.checkpoints
    )

    # Initialize trainer
    trainer = pl.Trainer(
        min_epochs=cfg.hparams.epoch_num,
        max_epochs=cfg.hparams.epoch_num,
        callbacks=checkpoints,
        accelerator="auto",
        devices="auto",
        logger=logger
    )

    # Train model
    trainer.fit(
        model=siamfc_model,
        datamodule=got10k_dm,
        ckpt_path=cfg.network.ckpt_path
    )

    # # Test model (bug)
    # trainer.test(
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