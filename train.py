import hydra
import torch
import pytorch_lightning as pl
from siamfc.siamfc import SiamFCNet
from siamfc.models import AlexNet, ContrastiveRandomWalkNet
from siamfc.data import ImageNetDataModule


AVAIL_GPUS = min(1, torch.cuda.device_count())


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # Initialize embedding network
    if cfg.model.architecture == "alexnet":
        embedding_net = AlexNet()
    elif cfg.model.architecture == "crwnet":
        embedding_net = ContrastiveRandomWalkNet()
    else:
        return ValueError("Invalid embedding network architecture specified.")
    
    # Initialize SiamFC network
    model = SiamFCNet(embedding_net)
    
    # Initialize data module
    dm = ImageNetDataModule(cfg.data.dir)
    dm.prepare_data()
    dm.setup(stage="fit")
    
    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=cfg.hparams.max_epochs,
        progress_bar_refresh_rate=20
    )
    
    # Train model
    trainer.fit(model, datamodule=dm)
    
    # Test model
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
