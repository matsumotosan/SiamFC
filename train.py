import hydra
import torch
from pytorch_lightning import Trainer
from siamfc import SiameseNet
from siamfc.models import AlexNet, ContrastiveRandomWalkNet
from siamfc.data import ImageNetDataModule


AVAIL_GPUS = min(1, torch.cuda.device_count())


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    
    # Initialize embedding network
    if cfg.model.architecture == "AlexNet":
        embedding_net = AlexNet()
    elif cfg.model.architecture == "CRWNet":
        embedding_net = ContrastiveRandomWalkNet()
    else:
        return ValueError("Invalid embedding network architecture specified.")
    
    # Initialize SiamFC network
    model = SiameseNet(embedding_net)
    
    # Initialize data module
    dm = ImageNetDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")
    
    # Initialize a trainer
    trainer = Trainer(
        max_epochs=cfg.hparams.max_epochs,
        progress_bar_refresh_rate=20
    )
    
    # Train model
    trainer.fit(model, datamodule=dm)
    
    # Test model
    trainer.test(datamodule=dm)


if __name__ == "__main__":
    main()
