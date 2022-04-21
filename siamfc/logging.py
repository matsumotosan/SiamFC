from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def setup_logger(logger="tensorboard", save_dir="logs", name=None):
    """Returns logger to be used in training and testing."""
    loggers = []
    if "tensorboard" in logger: 
        loggers.append(TensorBoardLogger(save_dir=save_dir, name=name))
    elif "wandb" in logger:
        loggers.append(WandbLogger(save_dir=save_dir, name=name))
    else:
        raise ValueError("Invalid logger specified.")

    return loggers